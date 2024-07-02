set shiftwidth=4
set tabstop=4
set expandtab

"""""""""""""""
" dein setup
"""""""""""""""
let $CACHE = expand('~/.cache')
if !isdirectory($CACHE)
  call mkdir($CACHE, 'p')
endif
if &runtimepath !~# '/dein.vim'
  let s:dein_dir = fnamemodify('dein.vim', ':p')
  if !isdirectory(s:dein_dir)
    let s:dein_dir = $CACHE .. '/dein/repos/github.com/Shougo/dein.vim'
    if !isdirectory(s:dein_dir)
      execute '!git clone https://github.com/Shougo/dein.vim' s:dein_dir
    endif
  endif
  execute 'set runtimepath^=' .. substitute(
        \ fnamemodify(s:dein_dir, ':p') , '[/\\]$', '', '')
endif

" dein scripts --------------------------------------------------
" Ward off unexpected things that your distro might have made, as
" well as sanely reset options when re-sourcing .vimrc
set nocompatible

" Set dein base path (required)
let s:dein_base = '~/.cache/dein/'

" Set dein source path (required)
let s:dein_src = '~/.cache/dein/repos/github.com/Shougo/dein.vim'

" Set dein runtime path (required)
execute 'set runtimepath+=' .. s:dein_src

" Call dein initialization (required)
call dein#begin(s:dein_base)
call dein#add(s:dein_src)

let s:toml = '~/.config/nvim/dein/dein.toml'
call dein#load_toml(s:toml, {'lazy':0})

"""""""""""""""""""""""
" Your plugins go here:
"""""""""""""""""""""""
" snippet
call dein#add('Shougo/deoplete.nvim')
if !has('nvim')
  call dein#add('roxma/nvim-yarp')
  call dein#add('roxma/vim-hug-neovim-rpc')
endif
let g:deoplete#enable_at_startup = 1
call dein#add('Shougo/neosnippet.vim')
call dein#add('Shougo/neosnippet-snippets')

" ddc
call dein#add('vim-denops/denops.vim')
call dein#add('Shougo/ddc.vim')
call dein#add('Shougo/ddc-around')
call dein#add('Shougo/ddc-mocword')
call dein#add('Shougo/ddc-matcher_head')
call dein#add('Shougo/ddc-sorter_rank')
call dein#add('Shougo/ddc-ui-pum')
call dein#add('Shougo/pum.vim')

" lsp
call dein#add('prabirshrestha/vim-lsp')
call dein#add('mattn/vim-lsp-settings')
call dein#add('shun/ddc-vim-lsp')

" syntax

" Finish dein initialization (equired)
call dein#end()

" Attempt to determine the type of a file based on its name and possibly its
" contents. Use this to allow intelligent auto-indenting for each filetype,
" and for plugins that are filetype specific.
filetype indent plugin on

" Enable syntax highlighting
if has('syntax')
  syntax on
endif

"""""""""""""""""""
" ddc setup
"""""""""""""""""""
call ddc#custom#patch_global('sources', ['around', 'mocword', 'vim-lsp'])
call ddc#custom#patch_global('sourceOptions', {
      \ 'around': {
      \     'mark': 'A',
      \     'minAutoCompleteLength': 3, 
      \ },
      \ 'vim-lsp': #{
      \     matchers: ['matcher_head'],
      \     mark: 'lsp',
      \ },
      \ 'mocword': #{
      \     mark: 'mocword',
      \     minAutoCompleteLength: 3, 
      \     isVolatile: v:true,
      \  },
      \ '_': {
      \   'matchers': ['matcher_head'],
      \   'sorters': ['sorter_rank']},
      \ })
call ddc#custom#patch_global(#{
            \   ui: 'pum',
            \   autoCompleteEvents: [
            \     'InsertEnter', 'TextChangedI', 'TextChangedP',
            \   ],
            \ })
call ddc#enable()

" disable linter by lsp
call lsp#disable_diagnostics_for_buffer()


" Uncomment if you want to install not-installed plugins on startup.
"if dein#check_install()
" call dein#install()
"endif
