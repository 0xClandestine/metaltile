use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Expr, Ident, ItemFn, Token, parse::ParseStream};

use crate::kernel::variants::{self, VariantsSpec};

/// Parsed arguments for the `#[bench]` attribute.
///
/// No `name` field — the bench name is always the (variant-renamed) function
/// name.  For a variants bench with `suffix = "int{BITS}"`, each variant's
/// function is renamed to e.g. `bench_foo_int4`, which becomes its bench name.
struct BenchAttr {
    /// Data types to exercise, e.g. `[f32, f16, bf16]`.
    dtypes: Vec<Ident>,
    /// Optional `|s: &BenchSetup| -> u64` closure overriding bytes-moved.
    bytes: Option<Expr>,
    /// Optional `|s: &BenchSetup| -> u64` closure overriding the FLOP count.
    flops: Option<Expr>,
    /// Optional `MetalRef { ... }` expression for reference comparison.
    metal_ref: Option<Expr>,
    /// Optional compile-time variant specialisation.
    variants: Option<VariantsSpec>,
}

impl syn::parse::Parse for BenchAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut dtypes = None;
        let mut bytes = None;
        let mut flops = None;
        let mut metal_ref = None;
        let mut variants_spec = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;

            if key == "variants" {
                let inner;
                syn::parenthesized!(inner in input);
                variants_spec = Some(inner.parse::<VariantsSpec>()?);
            } else {
                input.parse::<Token![=]>()?;

                if key == "dtypes" {
                    let content;
                    syn::bracketed!(content in input);
                    let list = content.parse_terminated(Ident::parse, Token![,])?;
                    dtypes = Some(list.into_iter().collect::<Vec<_>>());
                } else if key == "bytes" {
                    bytes = Some(input.parse::<Expr>()?);
                } else if key == "flops" {
                    flops = Some(input.parse::<Expr>()?);
                } else if key == "ref" {
                    metal_ref = Some(input.parse::<Expr>()?);
                } else {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown #[bench] key `{key}` — \
                             valid keys: dtypes, bytes, flops, ref, variants"
                        ),
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(BenchAttr {
            dtypes: dtypes.ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "#[bench] requires `dtypes = [f32, ...]`",
                )
            })?,
            bytes,
            flops,
            metal_ref,
            variants: variants_spec,
        })
    }
}

/// Map a dtype identifier string to its `DType::*` token stream.
pub(crate) fn dtype_token(ident: &str) -> TokenStream2 {
    match ident {
        "f32" => quote! { ::metaltile::core::DType::F32 },
        "f16" => quote! { ::metaltile::core::DType::F16 },
        "bf16" => quote! { ::metaltile::core::DType::BF16 },
        "i32" => quote! { ::metaltile::core::DType::I32 },
        "u32" => quote! { ::metaltile::core::DType::U32 },
        "i8" => quote! { ::metaltile::core::DType::I8 },
        "u8" => quote! { ::metaltile::core::DType::U8 },
        other => {
            let msg = format!("unknown dtype `{other}` in dtypes = [...]");
            quote! { compile_error!(#msg) }
        },
    }
}

/// Expand a single concrete `#[bench]` function into its `KernelBench` impl.
///
/// The bench name is always the function name.  For variants benches the
/// function has already been renamed to include the suffix before this is called.
fn expand_one(bench_attr: &BenchAttr, input_fn: ItemFn) -> TokenStream2 {
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let name_lit = syn::LitStr::new(&fn_name_str, fn_name.span());
    let impl_name = syn::Ident::new(&format!("__BenchImpl_{fn_name_str}"), fn_name.span());

    let dtype_tokens: Vec<TokenStream2> =
        bench_attr.dtypes.iter().map(|id| dtype_token(&id.to_string())).collect();

    let bytes_impl: TokenStream2 = match &bench_attr.bytes {
        Some(expr) => quote! {
            fn bytes_moved(
                &self,
                setup: &::metaltile::harness::bench::BenchSetup,
            ) -> u64 {
                (#expr)(setup)
            }
        },
        None => quote! {},
    };

    let flops_impl: TokenStream2 = match &bench_attr.flops {
        Some(expr) => quote! {
            fn flops(
                &self,
                setup: &::metaltile::core::bench::BenchSetup,
            ) -> ::std::option::Option<u64> {
                ::std::option::Option::Some((#expr)(setup))
            }
        },
        None => quote! {},
    };

    let metal_ref_impl: TokenStream2 = match &bench_attr.metal_ref {
        Some(expr) => quote! {
            fn reference_kernel(
                &self,
            ) -> ::std::option::Option<::metaltile::harness::bench::RefKernel> {
                ::std::option::Option::Some(#expr)
            }
        },
        None => quote! {},
    };

    let static_name = syn::Ident::new(&format!("__STATIC_{fn_name_str}"), fn_name.span());

    quote! {
        #input_fn

        #[allow(non_camel_case_types)]
        struct #impl_name;

        impl ::metaltile::harness::bench::KernelBench for #impl_name {
            fn name(&self) -> &str { #name_lit }

            fn dtypes(&self) -> &[::metaltile::core::DType] {
                &[#(#dtype_tokens),*]
            }

            fn setup(
                &self,
                dt: ::metaltile::core::DType,
            ) -> ::metaltile::harness::bench::BenchSetup {
                #fn_name(dt)
            }

            #bytes_impl
            #flops_impl
            #metal_ref_impl
        }

        #[allow(non_upper_case_globals)]
        static #static_name: #impl_name = #impl_name;
        ::metaltile::core::inventory::submit! {
            ::metaltile::harness::bench::KernelBenchEntry::new(&#static_name, file!())
        }
    }
}

/// Expand `#[bench(...)]` on a setup function.
///
/// The bench name is always derived from the function name.  For a variants
/// bench the function is renamed to include the suffix before registration, so
/// e.g. `bench_dequant_gemv` + `suffix = "int{BITS}"` → bench names
/// `bench_dequant_gemv_int2`, `bench_dequant_gemv_int4`, etc.
pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr2: proc_macro2::TokenStream = attr.into();
    let item2: proc_macro2::TokenStream = item.into();

    let bench_attr = match syn::parse2::<BenchAttr>(attr2) {
        Ok(a) => a,
        Err(e) => return e.into_compile_error().into(),
    };
    let input_fn = match syn::parse2::<ItemFn>(item2) {
        Ok(f) => f,
        Err(e) => return e.into_compile_error().into(),
    };

    if input_fn.sig.inputs.len() != 1 {
        return syn::Error::new_spanned(
            &input_fn.sig,
            "#[bench] setup function must take exactly one argument: `dt: DType`",
        )
        .into_compile_error()
        .into();
    }

    let Some(spec) = bench_attr.variants.as_ref() else {
        return expand_one(&bench_attr, input_fn).into();
    };

    let base_name = input_fn.sig.ident.to_string();
    let mut out = TokenStream2::new();

    for i in 0..spec.variant_count {
        let params_ordered: Vec<(String, i64)> =
            spec.params.iter().map(|(n, vals)| (n.clone(), vals[i])).collect();

        let variant_fn = match variants::substitute_fn(
            input_fn.clone(),
            &params_ordered,
            &base_name,
            &spec.suffix,
        ) {
            Ok(f) => f,
            Err(e) => return e.into_compile_error().into(),
        };

        out.extend(expand_one(&bench_attr, variant_fn));
    }

    out.into()
}
