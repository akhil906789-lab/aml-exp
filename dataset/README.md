# Dataset

**Name:** SMS Spam Collection (Almeida & Hidalgo, UCI Machine Learning Repository)
**Source:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection
**Licence:** listed on the UCI page as CC BY 4.0 (attribution required). Please re-check the licence on that page before redistributing.

| Item | Detail |
|------|--------|
| One row | one SMS message |
| Input feature | `message` (text) |
| Target label | `label` : `ham` (not spam) or `spam` |
| Records | 5,572 (4,825 ham, 747 spam) |

`spam.csv` in this folder is the same data converted to two columns (`label,message`).
If you delete it, download the original from the UCI link above and save it as
`dataset/spam.csv` with columns `label,message`.

Citation: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. ACM DocEng 2011.
