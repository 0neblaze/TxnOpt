# Literature and citation index

The public repository stores bibliographic metadata only. Publisher PDFs are
private local research inputs under ignored `document/literature/`; they are
not covered by Apache-2.0 and must not be committed or redistributed.

| Key | Reference | Research role | DOI |
|---|---|---|---|
| `Schneider2014EVRPTW` | Schneider, M., Stenger, A., & Goeke, D. (2014). The Electric Vehicle-Routing Problem with Time Windows and Recharging Stations. *Transportation Science*, 48(4), 500-520. | Core EVRP-TW model, benchmark instances, and BKS source | [10.1287/trsc.2013.0490](https://doi.org/10.1287/trsc.2013.0490) |
| `Hiermann2016ElectricFleet` | Hiermann, G., Puchinger, J., Ropke, S., & Hartl, R. F. (2016). The Electric Fleet Size and Mix Vehicle Routing Problem with Time Windows and Recharging Stations. *European Journal of Operational Research*, 252(3), 995-1018. | Fleet-size and heterogeneous-fleet comparison | [10.1016/j.ejor.2016.01.038](https://doi.org/10.1016/j.ejor.2016.01.038) |
| `KeskinCatay2016PartialRecharge` | Keskin, M., & Çatay, B. (2016). Partial Recharge Strategies for the Electric Vehicle Routing Problem with Time Windows. *Transportation Research Part C*, 65, 111-127. | Partial-recharge roadmap and large-instance BKS source | [10.1016/j.trc.2016.01.013](https://doi.org/10.1016/j.trc.2016.01.013) |
| `Montoya2017NonlinearCharging` | Montoya, A., Guéret, C., Mendoza, J. E., & Villegas, J. G. (2017). The Electric Vehicle Routing Problem with Nonlinear Charging Function. *Transportation Research Part B*, 103, 87-110. | Nonlinear-charging roadmap | [10.1016/j.trb.2017.02.004](https://doi.org/10.1016/j.trb.2017.02.004) |
| `Pelletier2016GoodsDistribution` | Pelletier, S., Jabali, O., & Laporte, G. (2016). Goods Distribution with Electric Vehicles: Review and Research Perspectives. *Transportation Science*, 50(1), 3-22. | Domain review and research framing | [10.1287/trsc.2015.0646](https://doi.org/10.1287/trsc.2015.0646) |
| `Desaulniers2016ExactEVRPTW` | Desaulniers, G., Errico, F., Irnich, S., & Schneider, M. (2016). Exact Algorithms for Electric Vehicle-Routing Problems with Time Windows. *Operations Research*, 64(6), 1388-1405. | Exact branch-price-and-cut benchmark and Stage 6 context | [10.1287/opre.2016.1535](https://doi.org/10.1287/opre.2016.1535) |
| `Jia2022BilevelACO` | Jia, Y.-H., Mei, Y., & Zhang, M. (2022). A Bilevel Ant Colony Optimization Algorithm for Capacitated Electric Vehicle Routing Problem. *IEEE Transactions on Cybernetics*, 52(10), 10855-10868. | Bilevel route/charging decomposition and fixed-route charging heuristics; out-of-family CEVRP reference only | [10.1109/TCYB.2021.3069942](https://doi.org/10.1109/TCYB.2021.3069942) |
| `Jia2022ConfidenceACO` | Jia, Y.-H., Mei, Y., & Zhang, M. (2022). Confidence-Based Ant Colony Optimization for Capacitated Electric Vehicle Routing Problem With Comparison of Different Encoding Schemes. *IEEE Transactions on Evolutionary Computation*, 26(6), 1394-1408. | Selective lower-level evaluation, encoding analysis, and candidate-ranking context; not a safe-screening rule | [10.1109/TEVC.2022.3144142](https://doi.org/10.1109/TEVC.2022.3144142) |
| `Feng2024BilevelHGA` | Feng, C.-T., Jia, Y.-H., Yang, Q., Chen, W.-N., & Jiang, H. (2024). A Bilevel Hybrid Genetic Algorithm for Capacitated Electric Vehicle Routing Problem. In *2024 IEEE Congress on Evolutionary Computation (CEC)* (pp. 1-8). IEEE. | Hybrid upper-level search and lower-level screening/enumeration; heuristic pipeline reference only | [10.1109/CEC60901.2024.10611987](https://doi.org/10.1109/CEC60901.2024.10611987) |

## TxnOpt research assessments

- [External methods and benchmark boundary](txnopt-external-methods-and-benchmarks-2026.md)
- [Publication gap audit](txnopt-publication-gap-audit-2026.md)
- [Provider-neutral cloud resource boundary](txnopt-cloud-resource-boundary-2026.md)
- [Windows salvage source audit](txnopt-windows-salvage-source-audit.md)

## Local file audit

The local collection contains nine unique cited works and two alternate PDF
copies:

- `03_Keskin_Catay_2016_Transportation_Research_Part_C_alternate.pdf`
  has the same DOI and identical normalized extracted text as the indexed
  Keskin-Çatay PDF.
- `04_Montoya_Gueret_Mendoza_Villegas_2017_Transportation_Research_Part_B_alternate.pdf`
  has the same DOI as the indexed Montoya PDF but different PDF pagination and
  packaging. It is retained locally as an alternate copy, not cited twice.

Every local PDF is recorded in `document/literature/SHA256SUMS`. The local
README identifies the publisher Version of Record status and copyright holder.
