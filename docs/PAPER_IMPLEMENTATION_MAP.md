# Paper Implementation Map

| Paper item | Repository implementation |
| --- | --- |
| Structured OHLCV, technical, macro features | `hd_tilag.features.technical`, `hd_tilag.data.preprocess` |
| Pearson, MIC, Lasso, RFE ensemble feature selection | `hd_tilag.features.selection.select_features` |
| FDS daily sentiment proportions | `hd_tilag.features.sentiment.aggregate_daily_sentiment` |
| Industry/style graph construction | `hd_tilag.features.graphs.build_relation_matrices` |
| Dual GCN graph embedding, Eq. (7)-(8) | `hd_tilag.models.components.DualGraphEmbedding` |
| Attention-Gated Unit, Eq. (9)-(14) | `hd_tilag.models.components.AttentionGatedUnit` |
| Inter-Layer Gating Unit, Eq. (15)-(16) | `hd_tilag.models.components.InterLayerGatingUnit` |
| TILAGNet / MS-TFAM | `hd_tilag.models.components.TILAGNet` |
| Output mapping and BCE loss | `hd_tilag.models.hd_tilag.HDTILAG`, `hd_tilag.training.trainer` |
| Accuracy, precision, recall, F1, CR, AR, SR, MDD | `hd_tilag.training.metrics` |
