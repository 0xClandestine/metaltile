// Force metaltile-std's codegen unit into this binary so all
// inventory::submit! kernel registrations (bench + test) are linked.
// The workspace release profile uses codegen-units=1 so all registrations
// land in a single object; this const reference pulls that object in.
const _: &() = &metaltile_std::__STD_LINK_ANCHOR;

fn main() {
    metaltile::runner::run(metaltile::runner::Args::from_env());
}
