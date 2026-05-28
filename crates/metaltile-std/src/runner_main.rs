fn main() {
    // Force metaltile-std's object file into the link so all inventory::submit!
    // kernel registrations (bench + test) are included in this binary.
    metaltile_std::__link_kernels();
    metaltile::runner::run(metaltile::runner::Args::from_env());
}
