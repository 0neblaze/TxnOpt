# Repository Instructions

These instructions apply to every Codex conversation whose working directory is
this repository or one of its subdirectories.

## Literature Recommendation Policy

- Codex recommends literature but does not obtain it. Do not access the user's
  university library account, publisher subscription, or institutional proxy,
  and do not download article PDFs.
- Recommend only formally published journal articles. Prioritise work from
  leading universities or research institutes and, when otherwise comparable,
  papers with higher citation impact.
- Verify that each recommendation has a formal journal record and that its title
  and DOI agree with the publisher or another authoritative bibliographic
  source. Do not recommend a technical report, working paper, preprint,
  postprint, accepted manuscript, or author manuscript as a journal article.
- Return literature recommendations only in the conversation as a Markdown
  table with exactly two columns: `Paper title` and `DOI`. Do not create or
  update repository files merely to store a recommendation list.
- The user is responsible for locating and downloading every recommended PDF.
  If a formal PDF is unavailable, provide no substitute file and wait for the
  user to supply the Version of Record (VOR).

## User-Supplied PDF Policy

- Only handle a literature PDF after the user has explicitly supplied it or
  identified its local path.
- Before accepting it as a VOR, verify the title, authors, journal, publication
  year, volume/issue, page range or article number, DOI, and visible publisher
  or journal version markers.
- Reject and remove from the formal literature collection any technical report,
  working paper, preprint, postprint, accepted manuscript, peer-reviewed
  manuscript, or author manuscript.
- Store verified PDFs only in the local Git-ignored literature directory. Use a
  filename containing the authors, year, and journal, and update the local
  literature index and SHA-256 checksums.
- Never commit journal PDFs, licence-protected files, access-controlled copies,
  or the local literature shortcut to GitHub.
