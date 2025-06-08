# Bioinformatics and Computational Biology

## Overview

Comprehensive guide to bioinformatics and computational biology, covering sequence analysis, genomics, proteomics, structural biology, and systems biology with practical examples using R, Python, and specialized tools.

## Table of Contents

- [Sequence Analysis](#sequence-analysis)
- [Genomics](#genomics)
- [Transcriptomics](#transcriptomics)
- [Proteomics](#proteomics)
- [Phylogenetics](#phylogenetics)
- [Structural Biology](#structural-biology)
- [Systems Biology](#systems-biology)
- [Database Resources](#database-resources)
- [Tools and Software](#tools-and-software)

## Sequence Analysis

### DNA/RNA Sequence Manipulation

```r
# Bioconductor packages for sequence analysis
library(Biostrings)
library(seqinr)
library(BSgenome)

# Create DNA sequences
dna_seq <- DNAString("ATGCGATCGTAGCTAGCTAGC")
rna_seq <- RNAString("AUGCGAUCGUAGCUAGCUAGC")

# Basic sequence operations
length(dna_seq)
complement(dna_seq)
reverseComplement(dna_seq)

# Transcription and translation
rna_from_dna <- RNAString(dna_seq)
protein <- translate(rna_from_dna)

# Reading FASTA files
sequences <- readDNAStringSet("sequences.fasta")
names(sequences)
width(sequences)  # Sequence lengths

# Sequence composition
letterFrequency(dna_seq, letters = c("A", "T", "G", "C"))
dinucleotideFrequency(dna_seq)

# GC content calculation
gc_content <- letterFrequency(dna_seq, letters = "GC", as.prob = TRUE)
```

### Sequence Alignment

```r
library(Biostrings)
library(msa)

# Pairwise alignment
seq1 <- DNAString("ACGTACGTACGT")
seq2 <- DNAString("ACGTACCGTACGT")

# Global alignment
global_align <- pairwiseAlignment(seq1, seq2, type = "global")
score(global_align)
writePairwiseAlignments(global_align)

# Local alignment
local_align <- pairwiseAlignment(seq1, seq2, type = "local")

# Multiple sequence alignment
sequences <- DNAStringSet(c(
  seq1 = "ACGTACGTACGT",
  seq2 = "ACGTACCGTACGT",
  seq3 = "ACGTACGTACCT"
))

msa_result <- msa(sequences)
print(msa_result)

# Phylogenetic tree from alignment
library(ape)
dist_matrix <- dist.dna(as.DNAbin(msa_result))
tree <- nj(dist_matrix)  # Neighbor-joining tree
plot(tree)
```

### Motif Finding

```r
library(Biostrings)
library(seqLogo)

# Pattern matching
pattern <- "TATA"
matches <- matchPattern(pattern, dna_seq)
start(matches)
end(matches)

# Regular expression patterns
tss_pattern <- "TATAAA"  # TATA box
matches <- vmatchPattern(tss_pattern, sequences)

# Create sequence logo
library(ggseqlogo)
sequences_char <- as.character(sequences)
ggseqlogo(sequences_char)

# Position weight matrix
pwm <- consensusMatrix(sequences)
seqLogo(pwm)
```

## Genomics

### Genome Annotation

```r
library(GenomicRanges)
library(GenomicFeatures)
library(rtracklayer)

# Create genomic ranges
gr <- GRanges(
  seqnames = c("chr1", "chr1", "chr2"),
  ranges = IRanges(start = c(100, 200, 300), 
                   end = c(150, 250, 350)),
  strand = c("+", "-", "+"),
  gene_id = c("Gene1", "Gene2", "Gene3")
)

# Genomic operations
width(gr)  # Feature widths
start(gr)  # Start positions
end(gr)    # End positions

# Overlaps
query <- GRanges("chr1", IRanges(120, 180))
overlaps <- findOverlaps(query, gr)

# Reading annotation files
# gtf <- import("annotation.gtf")
# bed <- import("regions.bed")

# Working with transcript databases
# txdb <- makeTxDbFromGFF("annotation.gtf")
# genes <- genes(txdb)
# transcripts <- transcripts(txdb)
# exons <- exons(txdb)
```

### Variant Analysis

```r
library(VariantAnnotation)
library(BSgenome.Hsapiens.UCSC.hg38)

# Read VCF files
# vcf <- readVcf("variants.vcf", "hg38")
# 
# # Extract variant information
# rowRanges(vcf)  # Genomic positions
# info(vcf)       # INFO field
# geno(vcf)       # Genotype data
# 
# # Predict variant effects
# library(VariantAnnotation)
# predictions <- predictCoding(vcf, txdb, seqSource = Hsapiens)

# Variant filtering
# Filter by quality, depth, etc.
variant_filter <- function(vcf, min_qual = 30, min_depth = 10) {
  qual_filter <- qual(vcf) >= min_qual
  depth_filter <- geno(vcf)$DP >= min_depth
  vcf[qual_filter & depth_filter]
}
```

### Copy Number Analysis

```r
library(DNAcopy)
library(GenomicRanges)

# Sample copy number data
cn_data <- data.frame(
  chromosome = rep(1:22, each = 100),
  position = rep(1:100, 22),
  log2ratio = rnorm(2200, 0, 0.3)
)

# Circular binary segmentation
cna_object <- CNA(cn_data$log2ratio, 
                  cn_data$chromosome, 
                  cn_data$position,
                  data.type = "logratio")

# Smooth and segment
smoothed <- smooth.CNA(cna_object)
segments <- segment(smoothed)

# Plot results
plot(segments, plot.type = "w")

# Call gains and losses
called_segments <- segments.p(segments)
```

## Transcriptomics

### RNA-seq Analysis

```r
library(DESeq2)
library(edgeR)
library(limma)

# Create sample data
counts_matrix <- matrix(
  rpois(20000, lambda = 10), 
  nrow = 2000, 
  ncol = 10
)
rownames(counts_matrix) <- paste0("Gene", 1:2000)
colnames(counts_matrix) <- paste0("Sample", 1:10)

# Sample metadata
coldata <- data.frame(
  condition = rep(c("Control", "Treatment"), each = 5),
  batch = rep(c("Batch1", "Batch2"), 5)
)

# DESeq2 analysis
dds <- DESeqDataSetFromMatrix(
  countData = counts_matrix,
  colData = coldata,
  design = ~ condition
)

# Preprocessing
dds <- DESeq(dds)

# Results
results <- results(dds)
summary(results)

# Volcano plot
library(ggplot2)
volcano_data <- data.frame(
  log2FC = results$log2FoldChange,
  pvalue = -log10(results$padj),
  significant = results$padj < 0.05 & abs(results$log2FoldChange) > 1
)

ggplot(volcano_data, aes(x = log2FC, y = pvalue, color = significant)) +
  geom_point() +
  labs(title = "Volcano Plot",
       x = "Log2 Fold Change",
       y = "-Log10 Adjusted P-value") +
  theme_minimal()

# Gene set enrichment analysis
library(clusterProfiler)
library(org.Hs.eg.db)

# Extract significant genes
sig_genes <- rownames(results)[which(results$padj < 0.05)]

# GO enrichment
ego <- enrichGO(gene = sig_genes,
                OrgDb = org.Hs.eg.db,
                ont = "BP",
                pAdjustMethod = "BH",
                readable = TRUE)

# KEGG pathway analysis
kegg <- enrichKEGG(gene = sig_genes,
                   organism = "hsa",
                   pvalueCutoff = 0.05)

# Visualization
dotplot(ego)
cnetplot(ego)
```

### Single-cell RNA-seq

```r
library(Seurat)
library(SingleCellExperiment)

# Create Seurat object
pbmc <- CreateSeuratObject(counts = counts_matrix)

# Quality control metrics
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")

# Visualization
VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"))

# Filtering
pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & 
               nFeature_RNA < 2500 & 
               percent.mt < 20)

# Normalization
pbmc <- NormalizeData(pbmc)
pbmc <- FindVariableFeatures(pbmc)
pbmc <- ScaleData(pbmc)

# Principal component analysis
pbmc <- RunPCA(pbmc)
DimPlot(pbmc, reduction = "pca")

# Clustering
pbmc <- FindNeighbors(pbmc)
pbmc <- FindClusters(pbmc, resolution = 0.5)

# UMAP
pbmc <- RunUMAP(pbmc, dims = 1:10)
DimPlot(pbmc, reduction = "umap", group.by = "seurat_clusters")

# Find marker genes
markers <- FindAllMarkers(pbmc, only.pos = TRUE)
top_markers <- markers %>% 
  group_by(cluster) %>% 
  top_n(n = 2, wt = avg_log2FC)
```

## Proteomics

### Mass Spectrometry Data Analysis

```r
library(MSnbase)
library(MSstats)

# Read mass spectrometry data
# mzml_files <- list.files(pattern = "\\.mzML$")
# raw_data <- readMSData(mzml_files, mode = "onDisk")

# Protein identification and quantification
# Create sample data
protein_data <- data.frame(
  ProteinName = rep(paste0("Protein", 1:100), each = 6),
  PeptideSequence = paste0("PEPTIDE", rep(1:600)),
  Condition = rep(c("Control", "Treatment"), each = 300),
  Replicate = rep(1:3, 200),
  Intensity = rlnorm(600, meanlog = 10, sdlog = 1)
)

# Statistical analysis with MSstats
# processed <- dataProcess(protein_data)
# comparison <- groupComparison(processed)

# Pathway analysis for proteins
library(ReactomePA)
significant_proteins <- sample(protein_data$ProteinName, 20)

# ReactomePA analysis
# pathway_result <- enrichPathway(gene = significant_proteins,
#                                pvalueCutoff = 0.05,
#                                readable = TRUE)
```

### Protein Structure Analysis

```r
library(bio3d)

# Read PDB structure
# pdb <- read.pdb("1HZX")
# 
# # Extract protein chains
# chain_a <- trim.pdb(pdb, chain = "A")
# 
# # Calculate secondary structure
# ss <- stride(pdb)
# 
# # Ramachandran plot
# tor <- torsion.pdb(pdb)
# ramachandran(tor)
# 
# # Distance matrix
# dm <- dm(pdb)
# 
# # Molecular dynamics simulation
# library(MDAnalysis)  # Would require Python interface
```

## Phylogenetics

### Phylogenetic Analysis

```r
library(ape)
library(phangorn)
library(ggtree)

# Create sample sequence data
sequences <- matrix(sample(c("A", "T", "G", "C"), 500, replace = TRUE),
                    nrow = 5, ncol = 100)
rownames(sequences) <- paste0("Species", 1:5)

# Convert to phylogenetic data format
phyDat_sequences <- phyDat(sequences)

# Distance-based methods
dist_matrix <- dist.hamming(phyDat_sequences)
nj_tree <- NJ(dist_matrix)
upgma_tree <- upgma(dist_matrix)

# Maximum likelihood
ml_tree <- pml(nj_tree, phyDat_sequences)
ml_tree <- optim.pml(ml_tree)

# Bootstrap analysis
bootstrap_trees <- bootstrap.pml(ml_tree, bs = 100)
consensus_tree <- consensus(bootstrap_trees)

# Visualization with ggtree
library(ggtree)
p <- ggtree(consensus_tree) + 
  geom_tiplab() + 
  geom_nodelab() +
  theme_tree2()

print(p)

# Add bootstrap values
p + geom_nodepoint(aes(color = as.numeric(label) > 70), size = 3)
```

### Molecular Evolution

```r
library(seqinr)
library(ape)

# Calculate evolutionary rates
# dN/dS analysis for protein-coding sequences
calculate_dnds <- function(seq1, seq2) {
  # Simplified calculation
  # In practice, use specialized tools like PAML
  synonymous_sites <- sample(1:100, 20)
  nonsynonymous_sites <- sample(1:100, 30)
  
  dN <- length(nonsynonymous_sites) / 100
  dS <- length(synonymous_sites) / 100
  
  return(dN / dS)
}

# Codon usage analysis
codon_usage <- function(sequence) {
  codons <- substring(sequence, seq(1, nchar(sequence)-2, 3), 
                     seq(3, nchar(sequence), 3))
  table(codons)
}
```

## Structural Biology

### Protein Structure Prediction

```r
library(bio3d)

# Secondary structure prediction
predict_ss <- function(sequence) {
  # Simplified prediction - use specialized tools like PSIPRED
  ss_pred <- sample(c("H", "E", "C"), nchar(sequence), 
                    replace = TRUE, prob = c(0.3, 0.2, 0.5))
  names(ss_pred) <- 1:nchar(sequence)
  return(ss_pred)
}

# Homology modeling workflow
homology_modeling <- function(target_seq, template_pdb) {
  # 1. Template identification (BLAST search)
  # 2. Sequence alignment
  # 3. Model building (MODELLER, SWISS-MODEL)
  # 4. Model validation
  
  cat("Homology modeling workflow:\n")
  cat("1. Template search completed\n")
  cat("2. Sequence alignment performed\n")
  cat("3. 3D model built\n")
  cat("4. Model validated\n")
}

# Molecular docking simulation
docking_analysis <- function(protein_pdb, ligand_mol) {
  # Simplified docking workflow
  # Use AutoDock, Vina, or similar tools
  
  results <- data.frame(
    pose = 1:10,
    binding_energy = rnorm(10, -8, 2),
    rmsd = runif(10, 0, 3)
  )
  
  return(results)
}
```

## Systems Biology

### Network Analysis

```r
library(igraph)
library(STRINGdb)
library(WGCNA)

# Create protein-protein interaction network
create_ppi_network <- function(proteins) {
  # Simulate PPI data
  n_proteins <- length(proteins)
  edges <- data.frame(
    from = sample(proteins, 100, replace = TRUE),
    to = sample(proteins, 100, replace = TRUE),
    weight = runif(100, 0, 1)
  )
  
  # Remove self-loops
  edges <- edges[edges$from != edges$to, ]
  
  # Create graph
  network <- graph_from_data_frame(edges, directed = FALSE)
  return(network)
}

# Network analysis
proteins <- paste0("Protein", 1:50)
ppi_network <- create_ppi_network(proteins)

# Network properties
degree_centrality <- degree(ppi_network)
betweenness_centrality <- betweenness(ppi_network)
closeness_centrality <- closeness(ppi_network)

# Community detection
communities <- cluster_louvain(ppi_network)
plot(communities, ppi_network)

# Gene co-expression network (WGCNA)
wgcna_analysis <- function(expression_matrix) {
  # Check data quality
  gsg <- goodSamplesGenes(expression_matrix)
  
  # Choose soft-thresholding power
  powers <- c(c(1:10), seq(from = 12, to = 20, by = 2))
  sft <- pickSoftThreshold(expression_matrix, powerVector = powers)
  
  # Construct network
  net <- blockwiseModules(expression_matrix,
                         power = sft$powerEstimate,
                         TOMType = "unsigned",
                         minModuleSize = 30)
  
  return(net)
}
```

### Metabolic Pathway Analysis

```r
library(pathview)
library(KEGGREST)

# KEGG pathway analysis
kegg_pathway_analysis <- function(gene_list, fold_changes) {
  # Map genes to KEGG pathways
  # pathways <- enrichKEGG(gene = gene_list,
  #                       organism = "hsa",
  #                       pvalueCutoff = 0.05)
  
  # Visualize pathway
  # pathview(gene.data = fold_changes,
  #         pathway.id = "04110",  # Cell cycle pathway
  #         species = "hsa")
  
  cat("KEGG pathway analysis completed\n")
}

# Flux balance analysis
fba_analysis <- function(stoichiometric_matrix, bounds) {
  # Simplified FBA implementation
  # Use specialized tools like COBRA toolbox
  
  # Objective function (maximize growth)
  objective <- rep(0, ncol(stoichiometric_matrix))
  objective[1] <- 1  # Growth reaction
  
  # Linear programming solution
  # solution <- lp("max", objective, stoichiometric_matrix, 
  #               "=", rep(0, nrow(stoichiometric_matrix)))
  
  cat("Flux balance analysis completed\n")
}
```

## Database Resources

### Major Biological Databases

```r
# NCBI databases
library(rentrez)

# Search PubMed
pubmed_search <- entrez_search(db = "pubmed", 
                              term = "CRISPR AND gene editing",
                              retmax = 100)

# Get sequence data from GenBank
# sequence_data <- entrez_fetch(db = "nucleotide", 
#                              id = "NM_000546",  # p53 mRNA
#                              rettype = "fasta")

# UniProt database access
library(UniProt.ws)
# up <- UniProt.ws(taxId = 9606)  # Human
# proteins <- select(up, keys = c("P04637"), 
#                   columns = c("GENENAME", "LENGTH", "SEQUENCE"))

# Ensembl database
library(biomaRt)
# ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
# genes <- getBM(attributes = c("ensembl_gene_id", "hgnc_symbol", 
#                              "chromosome_name", "start_position"),
#               mart = ensembl)
```

### Data Integration

```r
# Integrate multiple data types
integrate_omics_data <- function(genomics_data, transcriptomics_data, 
                               proteomics_data) {
  # Match samples across datasets
  common_samples <- Reduce(intersect, list(
    colnames(genomics_data),
    colnames(transcriptomics_data),
    colnames(proteomics_data)
  ))
  
  # Standardize and combine
  integrated_data <- list(
    genomics = genomics_data[, common_samples],
    transcriptomics = transcriptomics_data[, common_samples],
    proteomics = proteomics_data[, common_samples]
  )
  
  return(integrated_data)
}

# Multi-omics factor analysis
library(MOFA2)
# mofa_model <- create_mofa_from_matrix(integrated_data)
# mofa_model <- prepare_mofa(mofa_model)
# mofa_model <- run_mofa(mofa_model)
```

## Tools and Software

### Command Line Tools

```bash
# Sequence alignment tools
# BLAST
blastn -query query.fasta -db nt -out results.txt

# BWA alignment
bwa mem reference.fasta reads1.fastq reads2.fastq > aligned.sam

# SAMtools
samtools view -bS aligned.sam > aligned.bam
samtools sort aligned.bam -o sorted.bam
samtools index sorted.bam

# Variant calling with BCFtools
bcftools mpileup -f reference.fasta sorted.bam | \
bcftools call -mv -o variants.vcf

# Quality control with FastQC
fastqc raw_reads.fastq

# Assembly with SPAdes
spades.py -1 reads1.fastq -2 reads2.fastq -o assembly_output
```

### Python Integration

```python
# Biopython examples
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqUtils import GC

# Read FASTA file
sequences = SeqIO.parse("sequences.fasta", "fasta")
for record in sequences:
    print(f"ID: {record.id}, Length: {len(record.seq)}")
    print(f"GC content: {GC(record.seq):.2f}%")

# Sequence alignment
aligner = Align.PairwiseAligner()
alignments = aligner.align("ACGTACGT", "ACGTACCGT")
for alignment in alignments:
    print(alignment)

# Protein analysis with scikit-bio
import skbio
protein_seq = skbio.Protein("MKTVRQERLKSIVRIL")
print(f"Molecular weight: {protein_seq.molecular_weight()}")
```

### Workflow Management

```yaml
# Nextflow workflow example
process QUALITY_CONTROL {
    input:
    path fastq
    
    output:
    path "*.html"
    
    script:
    """
    fastqc ${fastq}
    """
}

process ALIGNMENT {
    input:
    path reference
    path reads
    
    output:
    path "*.bam"
    
    script:
    """
    bwa mem ${reference} ${reads} | samtools view -bS - > aligned.bam
    """
}

workflow {
    fastq_ch = Channel.fromPath("*.fastq")
    reference = file("reference.fasta")
    
    QUALITY_CONTROL(fastq_ch)
    ALIGNMENT(reference, fastq_ch)
}
```

---

*This bioinformatics guide provides a foundation for computational biology analysis. Each section can be expanded based on specific research needs and experimental designs.* 