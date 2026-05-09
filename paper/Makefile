MDPI_TEMPLATE_URL = https://res.mdpi.com/data/MDPI_template_ACS.zip?v=20250927
MDPI_TEMPLATE_FILE = mdpi-template.zip
OUTPUT_FILE = src/main.pdf

ifeq ($(shell uname),Darwin)
  OPEN = open
else
  OPEN = xdg-open
endif

.PHONY: build clean show install-mdpi-template remove-mdpi-template

build:
	@echo "Building the LaTeX document..."
	@cd src && latexmk -pdf main.tex

show:
	@echo "Opening the generated PDF..."
	@$(OPEN) $(OUTPUT_FILE)

clean:
	@echo "Removing all LaTeX auxiliary files..."
	@rm -f src/*.aux src/*.fdb_latexmk src/*.fls src/*.log src/*.out \
	  src/*.pdf src/*.gz indent.log src/*.bbl src/*.blg
	@echo "Removed all LaTeX auxiliary files..."

install-mdpi-template:
	@echo "Downloading MDPI LaTeX template..."
	@curl -L -o $(MDPI_TEMPLATE_FILE) $(MDPI_TEMPLATE_URL)
	@echo "Unzipping MDPI LaTeX template..."
	@unzip $(MDPI_TEMPLATE_FILE) -d src/mdpi
	@echo "Creating soft links to MDPI definition files..."
	@ln -s ${PWD}/src/mdpi/Definitions src/Definitions
	@echo "Cleaning up..."
	@rm mdpi-template.zip

remove-mdpi-template:
	@echo "Removing MDPI LaTeX template files..."
	@rm -rf src/mdpi
	@echo "Removing soft links to MDPI definition files..."
	@rm -f src/Definitions

paper.zip:
	@echo "Creating a zip archive of the paper..."
	@zip -r paper.zip src
	@echo "Created paper.zip archive."
