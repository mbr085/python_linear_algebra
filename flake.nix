{
  description = "Python environment using uv";

inputs = {
  nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
};

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python312;
      pythonPackages = pkgs.python312Packages;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          python
          pkgs.uv
          pkgs.gcc
          pkgs.stdenv
          pkgs.ghostscript
          pkgs.zlib
        ];

        shellHook = ''
          if [ ! -d .venv ]; then
            uv venv .venv
          fi
          export LD_LIBRARY_PATH="${pkgs.gcc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
          export PYTHONPATH="$PWD/src:$PYTHONPATH"  # 🔥 Fix: Add src/ to Python imports
          source .venv/bin/activate
        '';
      };
    };
}
