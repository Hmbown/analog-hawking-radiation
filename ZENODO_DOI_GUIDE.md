# How to Get a Zenodo DOI for Your Repository

## What is a Zenodo DOI?

A Zenodo DOI (Digital Object Identifier) makes your software citable in academic publications. Zenodo archives your GitHub repository and assigns a permanent DOI that researchers can reference.

## Step-by-Step Process

### 1. Create a GitHub Release

Before connecting to Zenodo, create a release on GitHub:

1. Go to your repository: https://github.com/hmbown/analog-hawking-radiation
2. Click "Releases" in the right sidebar
3. Click "Create a new release"
4. Fill in:
   - **Tag version**: `v0.1.0`
   - **Release title**: `Analog Hawking Radiation Framework v0.1.0`
   - **Description**: Brief summary of what this version includes
5. Click "Publish release"

### 2. Connect GitHub to Zenodo

1. Go to Zenodo: https://zenodo.org
2. Click "Log in" and select "Log in with GitHub"
3. Authorize Zenodo to access your GitHub account
4. Once logged in, go to: https://zenodo.org/account/settings/github/
5. Find your repository `hmbown/analog-hawking-radiation` in the list
6. Toggle the switch to "ON" to enable archiving

### 3. Create the Zenodo Archive

1. Go back to your GitHub repository
2. Create a new release (or re-release v0.1.0 if you just enabled Zenodo)
3. Zenodo will automatically archive this release within minutes
4. Go to https://zenodo.org/account/settings/github/repository/hmbown/analog-hawking-radiation
5. You should see your new release with a DOI badge

### 4. Get Your DOI

Once Zenodo creates the archive:

1. Click on the DOI badge or go to your Zenodo upload page
2. Your DOI will look like: `10.5281/zenodo.XXXXXXX`
3. Copy the full DOI URL: `https://doi.org/10.5281/zenodo.XXXXXXX`

### 5. Update Your Repository

Once you have the DOI, update these files:

**README.md** - Update the DOI badge (line 7):
```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

**CITATION.cff** - Add DOI field:
```yaml
doi: 10.5281/zenodo.XXXXXXX
```

**README.md Citation Section** - Update BibTeX:
```bibtex
@software{bown2025analog,
  author = {Bown, Hunter},
  title = {Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling},
  version = {0.1.0},
  year = {2025},
  url = {https://github.com/hmbown/analog-hawking-radiation},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## Additional Zenodo Configuration

### Add a .zenodo.json File (Optional)

Create `.zenodo.json` in your repository root for better metadata:

```json
{
  "title": "Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling",
  "description": "A computational framework for simulating analog Hawking radiation in laser-plasma systems through robust horizon detection, multi-beam configuration analysis, and radio-band detection feasibility assessment.",
  "creators": [
    {
      "name": "Bown, Hunter",
      "affiliation": "Independent Researcher",
      "orcid": "0000-0000-0000-0000"
    }
  ],
  "license": "MIT",
  "keywords": [
    "analog gravity",
    "Hawking radiation",
    "laser-plasma interactions",
    "horizon detection",
    "quantum field theory",
    "computational physics"
  ],
  "upload_type": "software",
  "access_right": "open"
}
```

Note: Replace `"orcid": "0000-0000-0000-0000"` with your actual ORCID if you have one, or remove that line.

## Concept DOI vs Version DOI

Zenodo creates two types of DOIs:

1. **Concept DOI**: Points to all versions (use this for general citations)
   - Example: `10.5281/zenodo.XXXXXXX`
   - Always resolves to the latest version

2. **Version-specific DOI**: Points to a specific release
   - Example: `10.5281/zenodo.XXXXXXX1` (note the extra digit)
   - Use this when citing a specific version in a paper

Typically, you want to display the **Concept DOI** in your README badge.

## Benefits of Having a DOI

1. **Citability**: Researchers can properly cite your software in academic papers
2. **Permanence**: Zenodo archives your code permanently (backed by CERN)
3. **Versioning**: Each release gets its own DOI while maintaining a concept DOI
4. **Discoverability**: Your software becomes searchable in academic databases
5. **Credit**: You get proper academic credit for your computational work

## Timeline

- GitHub Release → Zenodo archiving: ~5-15 minutes
- DOI assignment: Immediate upon archiving
- DOI activation: Immediate (can be cited right away)

## Troubleshooting

**Zenodo didn't create an archive**:
- Make sure the repository toggle is ON in Zenodo settings
- Try creating a new release (new tag name)
- Wait 15 minutes and check your Zenodo dashboard

**Don't see your repository in Zenodo**:
- Check that Zenodo has GitHub access permissions
- Try logging out and back in to Zenodo
- Refresh the GitHub repository list in Zenodo settings

**Want to update metadata after DOI creation**:
- Log into Zenodo, find your upload, click "Edit"
- You can update description, keywords, etc. without changing the DOI

## Resources

- Zenodo GitHub Guide: https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
- Zenodo Documentation: https://help.zenodo.org/
- Making Your Code Citable: https://guides.github.com/activities/citable-code/

## Next Steps

Once you have your DOI:
1. Update README.md badge
2. Update CITATION.cff
3. Add DOI to any papers or preprints about this work
4. Consider adding it to your GitHub repository description
5. Share the DOI when people ask how to cite your work

Your framework is already well-documented and validated - a Zenodo DOI will give it the academic credibility it deserves!
