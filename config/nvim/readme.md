# Install

## mocword for LSP

[Source](https://github.com/Shougo/ddc-source-mocword)

```sh
# .zshrc
chmod 764 mocword-x86_64-unknown-linux-musl
mv mocword-x86_64-unknown-linux-musl mocword /usr/local/bin/mocword
export MOCWORD_DATA=~/.cache/mocword.sqlite
```

## install deno
```sh
curl -fsSL https://deno.land/x/install/install.sh | sh
mv ~/.deno/bin/deno /usr/local/bin
```
## install pynvim
```sh
pip3 install pynvim
```

## install nerd font
```sh
wget https://github.com/ryanoasis/nerd-fonts/releases/download/v3.2.1/Hack.zip
unzip Hack.zip
sudo mv Hack.zip /usr/share/fonts/
sudo fc-cache -xvf
```
