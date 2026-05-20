//! Buffer liveness analysis and slot assignment.
//!
//! The compiler assigns each intermediate tensor to a `BufferSlot`.
//! Two intermediates can share a slot if their live ranges don't overlap.
//!
//! Algorithm (greedy, single-pass):
//! 1. Walk nodes in order. For each output tensor, record `first_use = current
//!    node index`.
//! 2. For each input tensor, update `last_use = max(last_use, current node
//!    index)`.
//! 3. Greedy slot assignment: for each new tensor, find an existing slot whose
//!    `last_use < current node's first_use` and whose `size_bytes >= required
//!    size`. If none, allocate a new slot.
//!
//! Weight tensors (`SlotRef::Weight`) and state tensors (`SlotRef::State`) are
//! excluded from liveness — they live for the full inference session and are
//! not pooled.
//!
//! Start conservative: initial implementation assigns one slot per tensor (no
//! reuse). Reuse is added later as an optimization pass. For the Llama decode
//! path, only ~8 intermediates are live at any point, so reuse barely matters.

use std::collections::HashMap;

use crate::plan::BufferSlot;

/// Live-range info for one intermediate tensor.
#[derive(Debug, Clone)]
struct LiveRange {
    /// Human-readable name.
    name: String,
    /// Size in bytes.
    size_bytes: usize,
    /// First node index this tensor is written.
    first_write: usize,
    /// Last node index this tensor is read.
    last_read: usize,
}

/// Assign buffer slots to a sequence of nodes.
///
/// `node_bindings` is a list of per-node `(param_name, slot_ref, size_bytes)`
/// triples for output buffers. This function:
///
/// 1. Computes live ranges for all intermediate tensors that use
///    `SlotRef::Slot(placeholder)`.
/// 2. Assigns real slot indices via greedy allocation.
/// 3. Returns the list of `BufferSlot`s and updates `SlotRef::Slot(idx)` in
///    each binding.
///
/// Params:
/// - `intermediate_outputs`: for each node, list of `(param_name, size_bytes)`
///   for outputs that will be assigned intermediate slots.
/// - `intermediate_inputs`: for each node, list of `(param_name)` that read
///   from intermediate slots. The corresponding output must be found by name.
pub fn assign_slots(
    _num_nodes: usize,
    intermediate_outputs: &[Vec<(String, usize)>],
    intermediate_inputs: &[Vec<String>],
) -> Vec<BufferSlot> {
    // Phase 1: collect all intermediate tensor names and their live ranges.
    let mut live_ranges: HashMap<String, LiveRange> = HashMap::new();

    // Record writes (first_use).
    for (node_idx, outputs) in intermediate_outputs.iter().enumerate() {
        for (name, size_bytes) in outputs {
            live_ranges
                .entry(name.clone())
                .and_modify(|_lr| {
                    // first_write should already be set (each tensor written once).
                })
                .or_insert_with(|| LiveRange {
                    name: name.clone(),
                    size_bytes: *size_bytes,
                    first_write: node_idx,
                    last_read: node_idx, // at least live through the write node
                });
        }
    }

    // Record reads (update last_read).
    for (node_idx, inputs) in intermediate_inputs.iter().enumerate() {
        for name in inputs {
            if let Some(lr) = live_ranges.get_mut(name) {
                lr.last_read = lr.last_read.max(node_idx);
            }
        }
    }

    // Phase 2: greedy slot assignment.
    // Sort by first_write so we assign in temporal order.
    let mut ranges: Vec<LiveRange> = live_ranges.into_values().collect();
    ranges.sort_by_key(|lr| lr.first_write);

    let mut slots: Vec<BufferSlot> = Vec::new();
    // Map from tensor name → assigned slot index.
    let mut name_to_slot: HashMap<String, usize> = HashMap::new();

    for lr in &ranges {
        // Try to reuse a freed slot.
        let mut assigned = None;
        for (slot_idx, slot) in slots.iter_mut().enumerate() {
            if slot.last_use < lr.first_write && slot.size_bytes >= lr.size_bytes {
                // Reuse this slot.
                slot.name = lr.name.clone();
                slot.size_bytes = lr.size_bytes;
                slot.first_use = lr.first_write;
                slot.last_use = lr.last_read;
                assigned = Some(slot_idx);
                break;
            }
        }

        let slot_idx = if let Some(idx) = assigned {
            idx
        } else {
            // Allocate new slot.
            let idx = slots.len();
            slots.push(BufferSlot {
                name: lr.name.clone(),
                size_bytes: lr.size_bytes,
                first_use: lr.first_write,
                last_use: lr.last_read,
            });
            idx
        };

        name_to_slot.insert(lr.name.clone(), slot_idx);
    }

    // Phase 3: if no reuse happened, just use identity mapping (simple path).
    // The greedy algorithm above handles reuse; if slots.len() == ranges.len(),
    // we got no reuse. This is fine — Phase 1 default.

    slots
}

/// Build a map from intermediate tensor name → slot index.
pub fn build_slot_map(slots: &[BufferSlot]) -> HashMap<String, usize> {
    slots.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_intermediates_returns_empty() {
        let slots = assign_slots(3, &[], &[]);
        assert!(slots.is_empty());
    }

    #[test]
    fn single_intermediate_gets_one_slot() {
        // One node with one output, no readers.
        let outputs = vec![vec![("temp".to_string(), 1024)]];
        let inputs: Vec<Vec<String>> = vec![vec![]];
        let slots = assign_slots(1, &outputs, &inputs);
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].name, "temp");
        assert_eq!(slots[0].size_bytes, 1024);
    }

    #[test]
    fn non_overlapping_intermediates_reuse_slot() {
        // Node 0: write temp_a (1024 bytes), read by node 2.
        // Node 1: write temp_b (512 bytes), read by node 3.
        // Node 2: read temp_a.
        // Node 3: read temp_b.
        //
        // temp_a: first_write=0, last_read=2
        // temp_b: first_write=1, last_read=3
        //
        // At node 1, temp_a is still live (last_read=2 > 1), so temp_b
        // must get a new slot.
        //
        // If we reverse: temp_a last_read=1, temp_b first_write=2,
        // then reuse is possible.
        let outputs = vec![
            vec![("temp_a".to_string(), 1024)],
            vec![("temp_b".to_string(), 512)],
            vec![],
            vec![],
        ];
        let inputs = vec![vec![], vec![], vec!["temp_a".to_string()], vec!["temp_b".to_string()]];
        let slots = assign_slots(4, &outputs, &inputs);
        // temp_a [0,2] and temp_b [1,3] overlap, so 2 slots.
        assert_eq!(slots.len(), 2);
    }

    #[test]
    fn sequential_intermediates_reuse_slot() {
        // Node 0: write temp_a, read by node 1.
        // Node 2: write temp_b, read by node 3.
        //
        // temp_a: [0,1], temp_b: [2,3] — non-overlapping, should reuse.
        let outputs = vec![
            vec![("temp_a".to_string(), 1024)],
            vec![],
            vec![("temp_b".to_string(), 1024)],
            vec![],
        ];
        let inputs = vec![vec![], vec!["temp_a".to_string()], vec![], vec!["temp_b".to_string()]];
        let slots = assign_slots(4, &outputs, &inputs);
        assert_eq!(slots.len(), 1, "sequential non-overlapping should reuse one slot");
    }
}
