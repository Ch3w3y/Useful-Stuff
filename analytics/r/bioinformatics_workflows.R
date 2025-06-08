# Bioinformatics Workflows in R
# ============================
# 
# Comprehensive template for bioinformatics analysis including:
# - Genomics and NGS data analysis
# - Proteomics and mass spectrometry
# - Single-cell RNA-seq analysis
# - Phylogenetic analysis
# - Systems biology and pathway analysis
# - Structural bioinformatics
#
# Author: Data Science Team
# Date: 2024

# Load Required Libraries
# -----------------------
# Core Bioconductor packages
library(BiocManager)
library(Biobase)
library(BiocGenerics)
library(S4Vectors)
library(IRanges)
library(GenomicRanges)
library(GenomicFeatures)
library(AnnotationDbi)
library(org.Hs.eg.db)      # Human annotation
library(TxDb.Hsapiens.UCSC.hg38.knownGene)

# NGS and genomics
library(Rsamtools)         # SAM/BAM file handling
library(GenomicAlignments) # Alignment data
library(rtracklayer)       # Import/export genomic data
library(VariantAnnotation) # Variant calling
library(BSgenome.Hsapiens.UCSC.hg38)
library(DNAcopy)           # Copy number analysis
library(CNTools)           # Copy number tools

# RNA-seq analysis
library(DESeq2)            # Differential expression
library(edgeR)             # RNA-seq analysis
library(limma)             # Linear models for microarrays
library(tximport)          # Import transcript-level estimates
library(GenomicFeatures)   # Genomic features
library(biomaRt)           # BioMart interface
library(clusterProfiler)   # Functional enrichment
library(DOSE)              # Disease ontology analysis
library(ReactomePA)        # Reactome pathway analysis
library(pathview)          # KEGG pathway visualization
library(enrichplot)        # Enrichment visualization

# Single-cell analysis
library(Seurat)            # Single-cell analysis
library(SingleCellExperiment)
library(scater)            # Single-cell QC and visualization
library(scran)             # Single-cell normalization
library(monocle3)          # Trajectory analysis
library(slingshot)         # Lineage inference
library(destiny)           # Diffusion maps
library(CellChat)          # Cell-cell communication

# Proteomics
library(MSstats)           # Statistical analysis of MS data
library(limma)             # Differential analysis
library(MSnbase)           # Mass spectrometry infrastructure
library(xcms)              # LC-MS data processing
library(CAMERA)            # Annotation of MS data
library(mzR)               # Raw MS data access

# Phylogenetics
library(ape)               # Phylogenetic analysis
library(phangorn)          # Phylogenetic reconstruction
library(ggtree)            # Tree visualization
library(treeio)            # Tree data I/O
library(phytools)          # Phylogenetic tools

# Systems biology
library(igraph)            # Network analysis
library(WGCNA)            # Weighted gene co-expression
library(STRINGdb)          # STRING database
library(RCy3)              # Cytoscape interface
library(graphite)          # Pathway graphs

# Structural bioinformatics
library(bio3d)             # Protein structure analysis
library(Rpdb)              # PDB file handling
library(rgl)               # 3D visualization

# Visualization and utilities
library(tidyverse)         # Data manipulation
library(ComplexHeatmap)    # Advanced heatmaps
library(circlize)          # Circular visualization
library(ggbio)             # Genomic data visualization
library(Gviz)              # Genomic data tracks
library(pheatmap)          # Pretty heatmaps
library(VennDiagram)       # Venn diagrams
library(UpSetR)            # UpSet plots
library(corrplot)          # Correlation plots

# Configuration
# -------------
options(stringsAsFactors = FALSE)
theme_set(theme_minimal())

# =================================================
# 1. GENOMICS AND NGS DATA ANALYSIS
# =================================================

#' Process RNA-seq Data with DESeq2
#' @param count_matrix Raw count matrix (genes x samples)
#' @param sample_info Sample metadata
#' @param design_formula Design formula for differential analysis
#' @return DESeq2 analysis results
analyze_rnaseq_deseq2 <- function(count_matrix, sample_info, design_formula) {
  
  cat("=== RNA-SEQ ANALYSIS WITH DESEQ2 ===\n")
  
  # Create DESeq2 dataset
  dds <- DESeqDataSetFromMatrix(
    countData = count_matrix,
    colData = sample_info,
    design = design_formula
  )
  
  # Filter low count genes
  keep <- rowSums(counts(dds)) >= 10
  dds <- dds[keep,]
  
  cat("Retained", sum(keep), "genes after filtering\n")
  
  # Run DESeq2 analysis
  dds <- DESeq(dds)
  
  # Quality control plots
  vsd <- vst(dds, blind = FALSE)
  
  # PCA plot
  pca_plot <- plotPCA(vsd, intgroup = c("condition")) +
    ggtitle("PCA Plot") +
    theme_minimal()
  
  print(pca_plot)
  
  # Sample distance heatmap
  sample_dists <- dist(t(assay(vsd)))
  sample_dist_matrix <- as.matrix(sample_dists)
  
  pheatmap(sample_dist_matrix,
           clustering_distance_rows = sample_dists,
           clustering_distance_cols = sample_dists,
           main = "Sample Distance Heatmap")
  
  # Get results
  res <- results(dds)
  res <- res[order(res$padj),]
  
  cat("\nDifferential Expression Summary:\n")
  cat("Upregulated genes (padj < 0.05, log2FC > 1):", 
      sum(res$padj < 0.05 & res$log2FoldChange > 1, na.rm = TRUE), "\n")
  cat("Downregulated genes (padj < 0.05, log2FC < -1):", 
      sum(res$padj < 0.05 & res$log2FoldChange < -1, na.rm = TRUE), "\n")
  
  # Volcano plot
  volcano_data <- data.frame(
    log2FoldChange = res$log2FoldChange,
    negLog10padj = -log10(res$padj),
    significant = res$padj < 0.05 & abs(res$log2FoldChange) > 1
  )
  
  volcano_plot <- ggplot(volcano_data, aes(x = log2FoldChange, y = negLog10padj)) +
    geom_point(aes(color = significant), alpha = 0.6) +
    scale_color_manual(values = c("grey", "red")) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed") +
    labs(title = "Volcano Plot", x = "log2 Fold Change", y = "-log10 adj. p-value") +
    theme_minimal()
  
  print(volcano_plot)
  
  return(list(
    dds = dds,
    results = res,
    vsd = vsd,
    plots = list(pca = pca_plot, volcano = volcano_plot)
  ))
}

#' Functional Enrichment Analysis
#' @param gene_list Vector of gene symbols
#' @param organism Organism (default: "hsa" for human)
#' @return Enrichment analysis results
functional_enrichment <- function(gene_list, organism = "hsa") {
  
  cat("=== FUNCTIONAL ENRICHMENT ANALYSIS ===\n")
  
  # Convert gene symbols to Entrez IDs
  gene_entrez <- bitr(gene_list, 
                      fromType = "SYMBOL",
                      toType = "ENTREZID", 
                      OrgDb = org.Hs.eg.db)
  
  # GO enrichment
  go_bp <- enrichGO(gene = gene_entrez$ENTREZID,
                    OrgDb = org.Hs.eg.db,
                    ont = "BP",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.2)
  
  go_mf <- enrichGO(gene = gene_entrez$ENTREZID,
                    OrgDb = org.Hs.eg.db,
                    ont = "MF",
                    pAdjustMethod = "BH",
                    pvalueCutoff = 0.05,
                    qvalueCutoff = 0.2)
  
  # KEGG pathway enrichment
  kegg <- enrichKEGG(gene = gene_entrez$ENTREZID,
                     organism = organism,
                     pvalueCutoff = 0.05)
  
  # Reactome pathway enrichment
  reactome <- enrichPathway(gene = gene_entrez$ENTREZID,
                           pvalueCutoff = 0.05,
                           readable = TRUE)
  
  # Visualization
  if (nrow(go_bp) > 0) {
    p1 <- dotplot(go_bp, showCategory = 20) + ggtitle("GO Biological Process")
    print(p1)
  }
  
  if (nrow(kegg) > 0) {
    p2 <- dotplot(kegg, showCategory = 20) + ggtitle("KEGG Pathways")
    print(p2)
  }
  
  if (nrow(reactome) > 0) {
    p3 <- dotplot(reactome, showCategory = 20) + ggtitle("Reactome Pathways")
    print(p3)
  }
  
  return(list(
    go_bp = go_bp,
    go_mf = go_mf,
    kegg = kegg,
    reactome = reactome
  ))
}

# =================================================
# 2. SINGLE-CELL RNA-SEQ ANALYSIS
# =================================================

#' Single-cell RNA-seq Analysis with Seurat
#' @param count_matrix Raw count matrix
#' @param project_name Project name
#' @return Seurat object with analysis results
analyze_single_cell <- function(count_matrix, project_name = "scRNA") {
  
  cat("=== SINGLE-CELL RNA-SEQ ANALYSIS ===\n")
  
  # Create Seurat object
  seurat_obj <- CreateSeuratObject(
    counts = count_matrix,
    project = project_name,
    min.cells = 3,
    min.features = 200
  )
  
  # Calculate mitochondrial gene percentage
  seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
  
  # QC visualization
  qc_plot <- VlnPlot(seurat_obj, 
                     features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
                     ncol = 3)
  print(qc_plot)
  
  # Feature scatter plots
  plot1 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "percent.mt")
  plot2 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
  print(plot1 + plot2)
  
  # Filter cells
  seurat_obj <- subset(seurat_obj, 
                       subset = nFeature_RNA > 200 & 
                               nFeature_RNA < 2500 & 
                               percent.mt < 20)
  
  # Normalization
  seurat_obj <- NormalizeData(seurat_obj)
  
  # Find variable features
  seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
  
  # Plot variable features
  top10 <- head(VariableFeatures(seurat_obj), 10)
  var_plot <- VariableFeaturePlot(seurat_obj)
  var_plot <- LabelPoints(plot = var_plot, points = top10, repel = TRUE)
  print(var_plot)
  
  # Scale data and run PCA
  all.genes <- rownames(seurat_obj)
  seurat_obj <- ScaleData(seurat_obj, features = all.genes)
  seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj))
  
  # PCA visualization
  pca_plot <- DimPlot(seurat_obj, reduction = "pca")
  print(pca_plot)
  
  # Elbow plot
  elbow_plot <- ElbowPlot(seurat_obj)
  print(elbow_plot)
  
  # Find neighbors and clusters
  seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10)
  seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)
  
  # Run UMAP
  seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)
  
  # Cluster visualization
  umap_plot <- DimPlot(seurat_obj, reduction = "umap", label = TRUE)
  print(umap_plot)
  
  # Find cluster markers
  cluster_markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, 
                                   logfc.threshold = 0.25)
  
  # Top markers per cluster
  top_markers <- cluster_markers %>%
    group_by(cluster) %>%
    slice_max(n = 5, order_by = avg_log2FC)
  
  cat("\nTop markers per cluster:\n")
  print(top_markers)
  
  # Heatmap of top markers
  top_genes <- top_markers %>% 
    group_by(cluster) %>% 
    top_n(n = 3, wt = avg_log2FC) %>%
    pull(gene)
  
  heatmap_plot <- DoHeatmap(seurat_obj, features = top_genes) + NoLegend()
  print(heatmap_plot)
  
  return(list(
    seurat_object = seurat_obj,
    markers = cluster_markers,
    top_markers = top_markers
  ))
}

#' Trajectory Analysis with Monocle3
#' @param seurat_obj Seurat object
#' @return Monocle3 analysis results
trajectory_analysis <- function(seurat_obj) {
  
  cat("=== TRAJECTORY ANALYSIS WITH MONOCLE3 ===\n")
  
  # Convert Seurat to cell_data_set
  cds <- as.cell_data_set(seurat_obj)
  cds <- cluster_cells(cds)
  
  # Learn trajectory graph
  cds <- learn_graph(cds)
  
  # Plot trajectory
  trajectory_plot <- plot_cells(cds, color_cells_by = "cluster", label_groups_by_cluster = FALSE,
                               label_leaves = FALSE, label_branch_points = FALSE)
  print(trajectory_plot)
  
  # Order cells in pseudotime (requires manual root selection)
  # cds <- order_cells(cds)
  
  return(cds)
}

# =================================================
# 3. PHYLOGENETIC ANALYSIS
# =================================================

#' Phylogenetic Tree Construction and Analysis
#' @param sequences DNA sequences (DNAbin object)
#' @return Phylogenetic analysis results
phylogenetic_analysis <- function(sequences) {
  
  cat("=== PHYLOGENETIC ANALYSIS ===\n")
  
  # Compute distance matrix
  dist_matrix <- dist.dna(sequences, model = "TN93")
  
  # Neighbor-joining tree
  nj_tree <- nj(dist_matrix)
  
  # Maximum likelihood tree
  fit <- pml(nj_tree, as.phyDat(sequences))
  fit_ml <- optim.pml(fit, model = "GTR", optGamma = TRUE)
  
  # Bootstrap support
  bs <- bootstrap.pml(fit_ml, bs = 100, optNni = TRUE)
  
  # Plot trees
  par(mfrow = c(2, 2))
  
  # NJ tree
  plot(nj_tree, main = "Neighbor-Joining Tree")
  
  # ML tree
  plot(fit_ml$tree, main = "Maximum Likelihood Tree")
  
  # ML tree with bootstrap
  plotBS(fit_ml$tree, bs, main = "ML Tree with Bootstrap")
  
  # Circular tree
  plot(fit_ml$tree, type = "fan", main = "Circular Tree")
  
  return(list(
    nj_tree = nj_tree,
    ml_tree = fit_ml$tree,
    bootstrap = bs,
    ml_fit = fit_ml
  ))
}

# =================================================
# 4. PROTEIN STRUCTURE ANALYSIS
# =================================================

#' Protein Structure Analysis
#' @param pdb_id PDB identifier
#' @return Structure analysis results
analyze_protein_structure <- function(pdb_id) {
  
  cat("=== PROTEIN STRUCTURE ANALYSIS ===\n")
  
  # Read PDB file
  pdb <- read.pdb(pdb_id)
  
  # Basic information
  cat("PDB ID:", pdb_id, "\n")
  cat("Resolution:", pdb$header$resolution, "Å\n")
  cat("Method:", pdb$header$technique, "\n")
  cat("Chains:", paste(unique(pdb$atom$chain), collapse = ", "), "\n")
  
  # Secondary structure prediction
  ss <- stride(pdb)
  
  # Ramachandran plot
  tor <- torsion.pdb(pdb)
  plot(tor$phi, tor$psi, xlab = "Phi", ylab = "Psi", 
       main = "Ramachandran Plot", pch = 20, col = "blue")
  
  # B-factor analysis
  ca_atoms <- atom.select(pdb, "calpha")
  plot(pdb$atom$b[ca_atoms$atom], type = "l", 
       xlab = "Residue Number", ylab = "B-factor",
       main = "B-factor Profile")
  
  # Distance matrix
  xyz <- pdb$atom[ca_atoms$atom, c("x", "y", "z")]
  dm <- dist(xyz)
  
  return(list(
    pdb = pdb,
    secondary_structure = ss,
    torsion_angles = tor,
    distance_matrix = dm
  ))
}

# =================================================
# 5. NETWORK ANALYSIS
# =================================================

#' Gene Co-expression Network Analysis
#' @param expr_matrix Expression matrix (genes x samples)
#' @return Network analysis results
coexpression_network <- function(expr_matrix) {
  
  cat("=== GENE CO-EXPRESSION NETWORK ANALYSIS ===\n")
  
  # Calculate correlation matrix
  cor_matrix <- cor(t(expr_matrix), use = "complete.obs")
  
  # Create adjacency matrix (soft thresholding)
  powers <- c(c(1:10), seq(from = 12, to = 20, by = 2))
  sft <- pickSoftThreshold(t(expr_matrix), powerVector = powers, verbose = 5)
  
  # Plot scale-free topology fit
  plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
       xlab = "Soft Threshold (power)", 
       ylab = "Scale Free Topology Model Fit, signed R^2",
       type = "n", main = paste("Scale independence"))
  text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
       labels = powers, cex = 0.9, col = "red")
  
  # Construct network
  soft_power <- sft$powerEstimate
  adjacency <- adjacency(t(expr_matrix), power = soft_power)
  
  # Calculate topological overlap
  TOM <- TOMsimilarity(adjacency)
  dissTOM <- 1 - TOM
  
  # Hierarchical clustering
  geneTree <- hclust(as.dist(dissTOM), method = "average")
  
  # Module identification
  modules <- cutreeDynamic(dendro = geneTree, distM = dissTOM,
                          deepSplit = 2, pamRespectsDendro = FALSE,
                          minClusterSize = 30)
  
  # Convert to colors
  module_colors <- labels2colors(modules)
  
  # Plot dendrogram with module colors
  plotDendroAndColors(geneTree, module_colors, "Module colors",
                     dendroLabels = FALSE, hang = 0.03,
                     addGuide = TRUE, guideHang = 0.05)
  
  # Module eigengenes
  MEList <- moduleEigengenes(t(expr_matrix), colors = module_colors)
  MEs <- MEList$eigengenes
  
  # Cluster module eigengenes
  ME_tree <- hclust(dist(t(MEs)), method = "average")
  plot(ME_tree, main = "Clustering of module eigengenes",
       xlab = "", sub = "")
  
  return(list(
    adjacency = adjacency,
    TOM = TOM,
    modules = modules,
    module_colors = module_colors,
    eigengenes = MEs,
    gene_tree = geneTree
  ))
}

#' Protein-Protein Interaction Network
#' @param gene_list Vector of gene symbols
#' @return PPI network analysis
ppi_network <- function(gene_list) {
  
  cat("=== PROTEIN-PROTEIN INTERACTION NETWORK ===\n")
  
  # STRING database interaction
  string_db <- STRINGdb$new(version = "11.5", species = 9606, 
                           score_threshold = 400, input_directory = "")
  
  # Map gene symbols to STRING IDs
  gene_mapped <- string_db$map(data.frame(gene = gene_list), "gene", removeUnmappedRows = TRUE)
  
  # Get interactions
  interactions <- string_db$get_interactions(gene_mapped$STRING_id)
  
  # Create igraph object
  g <- graph_from_data_frame(interactions, directed = FALSE)
  
  # Network statistics
  cat("Nodes:", vcount(g), "\n")
  cat("Edges:", ecount(g), "\n")
  cat("Density:", edge_density(g), "\n")
  cat("Clustering coefficient:", transitivity(g), "\n")
  
  # Centrality measures
  betweenness_cent <- betweenness(g)
  closeness_cent <- closeness(g)
  degree_cent <- degree(g)
  
  # Community detection
  communities <- cluster_louvain(g)
  
  # Visualization
  V(g)$size <- degree_cent * 2
  V(g)$color <- membership(communities)
  
  plot(g, vertex.label.cex = 0.7, vertex.label.color = "black",
       main = "Protein-Protein Interaction Network")
  
  # Network summary
  network_summary <- data.frame(
    gene = V(g)$name,
    degree = degree_cent,
    betweenness = betweenness_cent,
    closeness = closeness_cent,
    community = membership(communities)
  )
  
  return(list(
    graph = g,
    interactions = interactions,
    communities = communities,
    centrality = network_summary
  ))
}

# =================================================
# 6. EXAMPLE USAGE FUNCTIONS
# =================================================

#' Example: Complete RNA-seq Analysis Workflow
example_rnaseq_workflow <- function() {
  
  cat("=== EXAMPLE: RNA-SEQ ANALYSIS WORKFLOW ===\n")
  
  # Generate example data
  set.seed(42)
  n_genes <- 1000
  n_samples <- 12
  
  # Create count matrix
  base_expression <- rpois(n_genes, lambda = 100)
  count_matrix <- matrix(0, nrow = n_genes, ncol = n_samples)
  
  for (i in 1:n_samples) {
    # Add some noise and condition effects
    fold_changes <- rnorm(n_genes, mean = 1, sd = 0.3)
    if (i > 6) {  # Second condition
      # Some genes are differentially expressed
      de_genes <- sample(1:n_genes, size = 100)
      fold_changes[de_genes] <- rnorm(100, mean = 2, sd = 0.5)
    }
    count_matrix[, i] <- rpois(n_genes, lambda = base_expression * fold_changes)
  }
  
  # Add gene and sample names
  rownames(count_matrix) <- paste0("Gene_", 1:n_genes)
  colnames(count_matrix) <- paste0("Sample_", 1:n_samples)
  
  # Create sample info
  sample_info <- data.frame(
    sample_id = colnames(count_matrix),
    condition = rep(c("Control", "Treatment"), each = 6),
    batch = rep(c("A", "B"), times = 6)
  )
  rownames(sample_info) <- sample_info$sample_id
  
  # Run analysis
  rnaseq_results <- analyze_rnaseq_deseq2(
    count_matrix = count_matrix,
    sample_info = sample_info,
    design_formula = ~ condition
  )
  
  # Get significant genes
  sig_genes <- rownames(rnaseq_results$results)[
    which(rnaseq_results$results$padj < 0.05 & 
          abs(rnaseq_results$results$log2FoldChange) > 1)
  ]
  
  if (length(sig_genes) > 10) {
    # Functional enrichment (would need real gene symbols)
    cat("Significant genes found:", length(sig_genes), "\n")
    cat("Top 10 significant genes:\n")
    print(head(sig_genes, 10))
  }
  
  return(list(
    count_data = count_matrix,
    sample_data = sample_info,
    results = rnaseq_results,
    significant_genes = sig_genes
  ))
}

# Print available functions
if (FALSE) {  # Set to TRUE to run examples
  
  cat("Bioinformatics Workflows in R - Available Functions:\n")
  cat("=" * 60, "\n")
  cat("1. analyze_rnaseq_deseq2() - RNA-seq differential expression\n")
  cat("2. functional_enrichment() - GO/KEGG/Reactome enrichment\n")
  cat("3. analyze_single_cell() - scRNA-seq analysis with Seurat\n")
  cat("4. trajectory_analysis() - Pseudotime analysis\n")
  cat("5. phylogenetic_analysis() - Phylogenetic tree construction\n")
  cat("6. analyze_protein_structure() - Protein structure analysis\n")
  cat("7. coexpression_network() - Gene co-expression networks\n")
  cat("8. ppi_network() - Protein-protein interaction networks\n")
  cat("\nExample workflows:\n")
  cat("- example_rnaseq_workflow()\n")
  
  # Run example (uncomment to execute)
  # example_results <- example_rnaseq_workflow()
}

cat("Bioinformatics workflows template loaded successfully!\n")
cat("Use the functions above for comprehensive bioinformatics analysis.\n") 