project = "openfold3-accel"
author = "Eachan Johnson"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]
autodoc_mock_imports = ["openfold3"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/scbirlab/openfold3-accel",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
}
html_title = "openfold3-accel"
myst_enable_extensions = ["colon_fence"]
