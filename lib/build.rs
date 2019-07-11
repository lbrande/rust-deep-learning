fn main() {
    pkg_config::probe_library("mkl-dynamic-lp64-iomp").unwrap();
}
