# [Academic Paper (Nov, 2025)](https://github.com/aepw-uml/auda-docs/tree/main/academic-paper/20251128)

## Prerequisites

- Linux | macOS >= 26.1
- GUN Make >= 3.81
- Latexmk >= 4.86
- unzip >= 6.00
- TeX Live >= 2025 | MacTex (if on macOS)

## Installation

Run the following command to install [MDPI Template](https://www.mdpi.com/authors/latex):

```bash
make install-mdpi-template
```

## Build

> [!IMPORTANT]
> Please install MDPI template before you execute the `build` command.

Run the following command to build the `main.pdf`:

```bash
make build
```

If you have a `xdg-open` command (Linux) or a `open` command (macOS) on your system, you can run the following command to open the generated `main.pdf` after building it:

```bash

make show
```
