1. Treesitter &rarr; syntax highlighting

`TSInstall go` will install syntax highlighting for go. Use `go`, `gomod`, `gosum`, `gowork`

```Lua
local options = {
  ensure_installed = {
    "go",
    "lua",
    "bash",
    "gomod",
    "gosum",
    "gowork",
  },

  highlight = {
    enable = true,
    use_languagetree = true,
  },
  indent = { enabled = true }
}

require("nvim-treesitter.configs").setup(options)
```

