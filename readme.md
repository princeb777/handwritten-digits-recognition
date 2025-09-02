<!-- to build for windows -->
BINDGEN_EXTRA_CLANG_ARGS="--target=x86_64-w64-mingw32 -I/usr/x86_64-w64-mingw32/include" cargo build --release --target x86_64-pc-windows-gnu