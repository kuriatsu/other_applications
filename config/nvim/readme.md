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
