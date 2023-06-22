# %%
import jinja2 as j2
import git

# Generating license and description from the license.txt document, and the introduction paragraph of the readme.md
with open("LICENSE.txt") as f:
    license = " ".join(f.read().split("\n")).lstrip().rstrip()

with open("README.md") as f:
    readme = f.read().split("\n# ")
    for section in readme:
        if section.startswith("Introduction"):
            description = " ".join(section.split("\n")[1:]).lstrip().rstrip()

# Getting the package version.
repo = git.cmd.Git("./")
v = repo.describe("--tags").split("-")
version_string = f"{v[0]}.dev{v[1]}+{v[2]}"


# %% Applying the information to the template

templateLoader = j2.FileSystemLoader(searchpath="./")
templateEnv = j2.Environment(loader=templateLoader)
TEMPLATE_FILE = "pyproject.template"
template = templateEnv.get_template(TEMPLATE_FILE)
outputText = template.render(
    {
        "description": description,
        "license": license,
        "version": version_string
    }
)

# Writing the rendered template to the pyproject.toml file
with open("pyproject.toml", "w") as f:
    f.write(outputText)
