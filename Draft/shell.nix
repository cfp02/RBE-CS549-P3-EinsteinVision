{ pkgs ? import
    (builtins.fetchTarball {
      url = "https://github.com/nixos/nixpkgs/archive/19173a85eb60db6efe117523d4e597a182aae69b.tar.gz";
      sha256 = "1q1wvhm3cis5a7r8wjijfn6871irmh400wp25gqz6pc1kzymv9b2";
    }){}
}:
pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.texliveFull
  ];
}