# Genomics Data Processing Pipelines

Comprehensive guide for processing and analyzing genomic data using modern bioinformatics tools and cloud-native approaches.

## Table of Contents

- [Overview](#overview)
- [Core Technologies](#core-technologies)
- [Pipeline Architectures](#pipeline-architectures)
- [Workflow Management](#workflow-management)
- [Cloud Solutions](#cloud-solutions)
- [Quality Control](#quality-control)
- [Analysis Workflows](#analysis-workflows)

## Overview

Modern genomics pipelines require scalable, reproducible, and efficient processing of large-scale sequencing data. This guide covers end-to-end workflows from raw sequencing data to biological insights.

### Key Requirements

- **Scalability**: Handle terabytes of sequencing data
- **Reproducibility**: Containerized, versioned workflows
- **Compliance**: GDPR, HIPAA for clinical data
- **Performance**: Optimized for GPU/CPU intensive tasks
- **Interoperability**: Standard file formats and APIs

## Core Technologies

### Workflow Management Systems

#### 1. Nextflow - Domain Standard

```groovy
// Nextflow Pipeline for Whole Genome Sequencing
#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Parameters
params.reads = "data/*_{1,2}.fastq.gz"
params.genome = "/refs/hg38.fa"
params.outdir = "results"
params.threads = 8

// Include modules
include { FASTQC } from './modules/fastqc'
include { TRIMMOMATIC } from './modules/trimmomatic'
include { BWA_MEM } from './modules/bwa'
include { GATK_HAPLOTYPECALLER } from './modules/gatk'
include { ANNOVAR } from './modules/annovar'

workflow WGS_PIPELINE {
    
    // Input channel
    read_pairs = Channel
        .fromFilePairs(params.reads, checkIfExists: true)
        .map { name, files -> tuple(name, files[0], files[1]) }
    
    // Quality control
    FASTQC(read_pairs)
    
    // Trimming
    TRIMMOMATIC(read_pairs)
    
    // Alignment
    BWA_MEM(TRIMMOMATIC.out.trimmed_reads, params.genome)
    
    // Variant calling
    GATK_HAPLOTYPECALLER(BWA_MEM.out.bam, params.genome)
    
    // Annotation
    ANNOVAR(GATK_HAPLOTYPECALLER.out.vcf)
    
    // Collect outputs
    outputs = ANNOVAR.out.annotated
        .collect()
        .map { files -> 
            publishDir params.outdir, mode: 'copy'
            files
        }
}

workflow {
    WGS_PIPELINE()
}
```

#### 2. Snakemake - Python-based Workflows

```python
# Snakemake workflow for RNA-seq analysis
import pandas as pd

# Configuration
configfile: "config.yaml"

# Sample information
samples = pd.read_csv(config["samples"], sep="\t").set_index("sample", drop=False)

rule all:
    input:
        "results/multiqc_report.html",
        expand("results/counts/{sample}.counts", sample=samples.index),
        "results/deseq2/differential_expression.csv"

rule fastqc:
    input:
        "data/{sample}.fastq.gz"
    output:
        html="results/fastqc/{sample}_fastqc.html",
        zip="results/fastqc/{sample}_fastqc.zip"
    conda:
        "envs/fastqc.yaml"
    shell:
        "fastqc {input} -o results/fastqc/"

rule trim_reads:
    input:
        "data/{sample}.fastq.gz"
    output:
        "results/trimmed/{sample}_trimmed.fastq.gz"
    conda:
        "envs/trimmomatic.yaml"
    shell:
        """
        trimmomatic SE {input} {output} \
        ILLUMINACLIP:adapters.fa:2:30:10 \
        LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36
        """

rule align_star:
    input:
        reads="results/trimmed/{sample}_trimmed.fastq.gz",
        index=config["star_index"]
    output:
        "results/aligned/{sample}Aligned.sortedByCoord.out.bam"
    conda:
        "envs/star.yaml"
    threads: 8
    shell:
        """
        STAR --genomeDir {input.index} \
             --readFilesIn {input.reads} \
             --readFilesCommand zcat \
             --runThreadN {threads} \
             --outFileNamePrefix results/aligned/{wildcards.sample} \
             --outSAMtype BAM SortedByCoordinate
        """

rule count_features:
    input:
        bam="results/aligned/{sample}Aligned.sortedByCoord.out.bam",
        gtf=config["annotation"]
    output:
        "results/counts/{sample}.counts"
    conda:
        "envs/subread.yaml"
    shell:
        """
        featureCounts -a {input.gtf} -o {output} {input.bam}
        """

rule deseq2_analysis:
    input:
        counts=expand("results/counts/{sample}.counts", sample=samples.index),
        metadata=config["samples"]
    output:
        "results/deseq2/differential_expression.csv"
    conda:
        "envs/deseq2.yaml"
    script:
        "scripts/deseq2_analysis.R"

rule multiqc:
    input:
        expand("results/fastqc/{sample}_fastqc.zip", sample=samples.index),
        expand("results/aligned/{sample}Aligned.sortedByCoord.out.bam", sample=samples.index)
    output:
        "results/multiqc_report.html"
    conda:
        "envs/multiqc.yaml"
    shell:
        "multiqc results/ -o results/"
```

### Containerization

#### Docker Images for Bioinformatics

```dockerfile
# Multi-stage Dockerfile for genomics tools
FROM ubuntu:22.04 as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    python3 \
    python3-pip \
    r-base \
    openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

# Install bioinformatics tools
FROM base as tools

# Install BWA
RUN wget https://github.com/lh3/bwa/releases/download/v0.7.17/bwa-0.7.17.tar.bz2 && \
    tar -xvf bwa-0.7.17.tar.bz2 && \
    cd bwa-0.7.17 && \
    make && \
    cp bwa /usr/local/bin/

# Install SAMtools
RUN wget https://github.com/samtools/samtools/releases/download/1.17/samtools-1.17.tar.bz2 && \
    tar -xvf samtools-1.17.tar.bz2 && \
    cd samtools-1.17 && \
    ./configure && \
    make && \
    make install

# Install GATK
RUN wget https://github.com/broadinstitute/gatk/releases/download/4.4.0.0/gatk-4.4.0.0.zip && \
    unzip gatk-4.4.0.0.zip && \
    mv gatk-4.4.0.0 /opt/gatk

# Install Python packages
RUN pip3 install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    biopython \
    pysam \
    pyvcf

# Install R packages
RUN R -e "install.packages(c('DESeq2', 'ggplot2', 'pheatmap', 'BiocManager'), repos='http://cran.r-project.org')"
RUN R -e "BiocManager::install(c('GenomicFeatures', 'AnnotationDbi', 'org.Hs.eg.db'))"

# Set up environment
ENV PATH="/opt/gatk:$PATH"
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

WORKDIR /workspace
```

## Pipeline Architectures

### Cloud-Native Genomics Pipeline

```yaml
# Kubernetes CRD for Genomics Workflows
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: genomicsworkflows.bio.example.com
spec:
  group: bio.example.com
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              pipeline:
                type: string
                enum: ["wgs", "wes", "rnaseq", "chipseq"]
              samples:
                type: array
                items:
                  type: object
                  properties:
                    id: { type: string }
                    fastq_r1: { type: string }
                    fastq_r2: { type: string }
              reference:
                type: object
                properties:
                  genome: { type: string }
                  annotation: { type: string }
              resources:
                type: object
                properties:
                  cpu: { type: string }
                  memory: { type: string }
                  storage: { type: string }
          status:
            type: object
            properties:
              phase: { type: string }
              startTime: { type: string }
              completionTime: { type: string }
              outputs: { type: object }
  scope: Namespaced
  names:
    plural: genomicsworkflows
    singular: genomicsworkflow
    kind: GenomicsWorkflow

---
# Example Genomics Workflow
apiVersion: bio.example.com/v1
kind: GenomicsWorkflow
metadata:
  name: cancer-wgs-analysis
spec:
  pipeline: "wgs"
  samples:
    - id: "tumor_sample_1"
      fastq_r1: "gs://genomics-bucket/tumor_1_R1.fastq.gz"
      fastq_r2: "gs://genomics-bucket/tumor_1_R2.fastq.gz"
    - id: "normal_sample_1"
      fastq_r1: "gs://genomics-bucket/normal_1_R1.fastq.gz"
      fastq_r2: "gs://genomics-bucket/normal_1_R2.fastq.gz"
  reference:
    genome: "gs://genomics-bucket/refs/hg38.fa"
    annotation: "gs://genomics-bucket/refs/gencode.v44.gtf"
  resources:
    cpu: "32"
    memory: "128Gi"
    storage: "1Ti"
```

### Distributed Processing with Spark

```python
# PySpark for large-scale variant analysis
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import glow

# Initialize Spark with Glow (genomics extensions)
spark = SparkSession.builder \
    .appName("VariantAnalysis") \
    .config("spark.jars.packages", "io.projectglow:glow-spark3_2.12:1.2.1") \
    .getOrCreate()

# Register Glow functions
glow.register(spark)

# Define schema for VCF data
vcf_schema = StructType([
    StructField("chrom", StringType()),
    StructField("pos", LongType()),
    StructField("id", StringType()),
    StructField("ref", StringType()),
    StructField("alt", ArrayType(StringType())),
    StructField("qual", DoubleType()),
    StructField("filter", ArrayType(StringType())),
    StructField("info", MapType(StringType(), StringType())),
    StructField("genotypes", ArrayType(StructType([
        StructField("sampleId", StringType()),
        StructField("GT", StringType()),
        StructField("AD", ArrayType(IntegerType())),
        StructField("DP", IntegerType())
    ])))
])

def process_vcf_files(vcf_paths):
    """Process multiple VCF files for population analysis."""
    
    # Read VCF files
    df = spark.read \
        .format("vcf") \
        .option("includeSampleIds", True) \
        .load(vcf_paths)
    
    # Quality filtering
    filtered_df = df.filter(
        (col("qual") > 30) & 
        (size(col("filter")) == 0)  # PASS variants only
    )
    
    # Calculate allele frequencies
    af_df = filtered_df \
        .select(
            "chrom", "pos", "ref", "alt",
            explode("genotypes").alias("genotype")
        ) \
        .groupBy("chrom", "pos", "ref", "alt") \
        .agg(
            count("*").alias("total_samples"),
            sum(when(col("genotype.GT") == "1/1", 2)
                .when(col("genotype.GT").isin(["0/1", "1/0"]), 1)
                .otherwise(0)).alias("alt_allele_count")
        ) \
        .withColumn("allele_frequency", 
                   col("alt_allele_count") / (col("total_samples") * 2))
    
    # Annotate variants
    annotated_df = af_df.select(
        "chrom", "pos", "ref", "alt", "allele_frequency",
        glow.lift_over_coordinates(
            col("chrom"), col("pos"), lit("hg19"), lit("hg38")
        ).alias("hg38_coords")
    )
    
    return annotated_df

def gwas_analysis(genotype_df, phenotype_df):
    """Perform genome-wide association study."""
    
    # Join genotype and phenotype data
    joined_df = genotype_df.join(phenotype_df, "sampleId")
    
    # Calculate association statistics
    gwas_results = joined_df \
        .groupBy("chrom", "pos", "ref", "alt") \
        .agg(
            glow.logistic_regression_gwas(
                col("genotype_states"),
                col("phenotype"),
                col("covariates")
            ).alias("stats")
        ) \
        .select(
            "chrom", "pos", "ref", "alt",
            col("stats.beta").alias("effect_size"),
            col("stats.pValue").alias("p_value"),
            col("stats.stdError").alias("standard_error")
        ) \
        .filter(col("p_value") < 0.05)  # Significant associations
    
    return gwas_results

# Example usage
vcf_paths = ["gs://genomics-data/cohort/*.vcf.gz"]
results = process_vcf_files(vcf_paths)

# Save results
results.write \
    .mode("overwrite") \
    .option("compression", "gzip") \
    .parquet("gs://results/variant_analysis/")
```

## Cloud Solutions

### AWS Genomics Workflows

```yaml
# AWS Batch job definition for genomics
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Genomics pipeline using AWS Batch and S3'

Parameters:
  VpcId:
    Type: AWS::EC2::VPC::Id
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>

Resources:
  # Batch Compute Environment
  GenomicsComputeEnvironment:
    Type: AWS::Batch::ComputeEnvironment
    Properties:
      Type: MANAGED
      State: ENABLED
      ComputeResources:
        Type: EC2
        MinvCpus: 0
        MaxvCpus: 1000
        DesiredvCpus: 0
        InstanceTypes: 
          - c5.large
          - c5.xlarge
          - c5.2xlarge
          - r5.large
          - r5.xlarge
        Ec2Configuration:
          - ImageType: ECS_AL2
        Subnets: !Ref SubnetIds
        SecurityGroupIds:
          - !Ref GenomicsSecurityGroup
        InstanceRole: !GetAtt InstanceProfile.Arn
        
  # Job Queue
  GenomicsJobQueue:
    Type: AWS::Batch::JobQueue
    Properties:
      State: ENABLED
      Priority: 1
      ComputeEnvironmentOrder:
        - Order: 1
          ComputeEnvironment: !Ref GenomicsComputeEnvironment

  # Job Definition for GATK
  GATKJobDefinition:
    Type: AWS::Batch::JobDefinition
    Properties:
      Type: container
      ContainerProperties:
        Image: broadinstitute/gatk:4.4.0.0
        Vcpus: 8
        Memory: 32768
        JobRoleArn: !GetAtt BatchExecutionRole.Arn
        Environment:
          - Name: AWS_DEFAULT_REGION
            Value: !Ref AWS::Region
        MountPoints:
          - SourceVolume: tmp
            ContainerPath: /tmp
        Volumes:
          - Name: tmp
            Host:
              SourcePath: /tmp

  # Lambda function for workflow orchestration
  GenomicsOrchestrator:
    Type: AWS::Lambda::Function
    Properties:
      Runtime: python3.9
      Handler: index.lambda_handler
      Role: !GetAtt LambdaExecutionRole.Arn
      Code:
        ZipFile: |
          import boto3
          import json
          
          def lambda_handler(event, context):
              batch = boto3.client('batch')
              
              # Extract parameters
              sample_id = event['sample_id']
              fastq_r1 = event['fastq_r1']
              fastq_r2 = event['fastq_r2']
              reference = event['reference']
              
              # Submit GATK job
              response = batch.submit_job(
                  jobName=f'gatk-{sample_id}',
                  jobQueue='GenomicsJobQueue',
                  jobDefinition='GATKJobDefinition',
                  parameters={
                      'sample_id': sample_id,
                      'fastq_r1': fastq_r1,
                      'fastq_r2': fastq_r2,
                      'reference': reference,
                      'output_bucket': 's3://genomics-results/'
                  }
              )
              
              return {
                  'statusCode': 200,
                  'body': json.dumps({
                      'jobId': response['jobId'],
                      'message': f'Job submitted for sample {sample_id}'
                  })
              }

  # Step Functions state machine
  GenomicsStateMachine:
    Type: AWS::StepFunctions::StateMachine
    Properties:
      DefinitionString: !Sub |
        {
          "Comment": "Genomics processing pipeline",
          "StartAt": "QualityControl",
          "States": {
            "QualityControl": {
              "Type": "Task",
              "Resource": "${GenomicsOrchestrator}",
              "Parameters": {
                "stage": "fastqc",
                "sample_id.$": "$.sample_id",
                "fastq_r1.$": "$.fastq_r1",
                "fastq_r2.$": "$.fastq_r2"
              },
              "Next": "Alignment"
            },
            "Alignment": {
              "Type": "Task",
              "Resource": "${GenomicsOrchestrator}",
              "Parameters": {
                "stage": "bwa_mem",
                "sample_id.$": "$.sample_id",
                "fastq_r1.$": "$.fastq_r1",
                "fastq_r2.$": "$.fastq_r2",
                "reference.$": "$.reference"
              },
              "Next": "VariantCalling"
            },
            "VariantCalling": {
              "Type": "Task",
              "Resource": "${GenomicsOrchestrator}",
              "Parameters": {
                "stage": "gatk_haplotypecaller",
                "sample_id.$": "$.sample_id",
                "bam_file.$": "$.bam_file",
                "reference.$": "$.reference"
              },
              "End": true
            }
          }
        }
      RoleArn: !GetAtt StepFunctionsRole.Arn
```

### Google Cloud Life Sciences

```python
# Google Cloud Life Sciences API for genomics pipelines
from google.cloud import lifesciences_v2beta
from google.cloud.lifesciences_v2beta.types import Pipeline, Action

def create_genomics_pipeline():
    """Create a genomics processing pipeline using Google Cloud Life Sciences."""
    
    client = lifesciences_v2beta.WorkflowsV2BetaClient()
    
    # Define pipeline actions
    fastqc_action = Action(
        container_name="fastqc",
        image_uri="biocontainers/fastqc:v0.11.9_cv8",
        commands=[
            "/bin/bash", "-c",
            "fastqc ${FASTQ_R1} ${FASTQ_R2} -o ${OUTPUT_DIR}"
        ],
        environment={
            "FASTQ_R1": "/mnt/data/sample_R1.fastq.gz",
            "FASTQ_R2": "/mnt/data/sample_R2.fastq.gz",
            "OUTPUT_DIR": "/mnt/output/fastqc/"
        }
    )
    
    bwa_action = Action(
        container_name="bwa",
        image_uri="biocontainers/bwa:v0.7.17_cv1",
        commands=[
            "/bin/bash", "-c",
            """
            bwa mem -t 8 ${REFERENCE} ${FASTQ_R1} ${FASTQ_R2} | \
            samtools view -Sb - | \
            samtools sort -o ${OUTPUT_DIR}/aligned.bam -
            samtools index ${OUTPUT_DIR}/aligned.bam
            """
        ],
        environment={
            "REFERENCE": "/mnt/refs/hg38.fa",
            "FASTQ_R1": "/mnt/data/sample_R1.fastq.gz",
            "FASTQ_R2": "/mnt/data/sample_R2.fastq.gz",
            "OUTPUT_DIR": "/mnt/output/alignment/"
        }
    )
    
    gatk_action = Action(
        container_name="gatk",
        image_uri="broadinstitute/gatk:4.4.0.0",
        commands=[
            "/bin/bash", "-c",
            """
            gatk HaplotypeCaller \
                -R ${REFERENCE} \
                -I ${INPUT_BAM} \
                -O ${OUTPUT_DIR}/variants.vcf.gz
            """
        ],
        environment={
            "REFERENCE": "/mnt/refs/hg38.fa",
            "INPUT_BAM": "/mnt/output/alignment/aligned.bam",
            "OUTPUT_DIR": "/mnt/output/variants/"
        }
    )
    
    # Create pipeline
    pipeline = Pipeline(
        actions=[fastqc_action, bwa_action, gatk_action],
        resources={
            "regions": ["us-central1"],
            "virtual_machine": {
                "machine_type": "n1-standard-8",
                "boot_disk_size_gb": 100,
                "preemptible": True
            }
        },
        environment={
            "INPUT_BUCKET": "gs://genomics-input/",
            "OUTPUT_BUCKET": "gs://genomics-output/"
        }
    )
    
    return pipeline

def run_genomics_workflow(sample_list):
    """Run genomics workflow for multiple samples."""
    
    client = lifesciences_v2beta.WorkflowsV2BetaClient()
    
    for sample in sample_list:
        pipeline = create_genomics_pipeline()
        
        # Customize for sample
        pipeline.environment.update({
            "SAMPLE_ID": sample["id"],
            "FASTQ_R1": sample["fastq_r1"],
            "FASTQ_R2": sample["fastq_r2"]
        })
        
        # Submit pipeline
        operation = client.run_pipeline(
            parent=f"projects/{PROJECT_ID}/locations/us-central1",
            pipeline=pipeline
        )
        
        print(f"Started pipeline for sample {sample['id']}: {operation.name}")
```

## Quality Control

### Comprehensive QC Pipeline

```python
# Comprehensive quality control for genomics data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import pysam

class GenomicsQC:
    """Comprehensive quality control for genomics pipelines."""
    
    def __init__(self, sample_id):
        self.sample_id = sample_id
        self.qc_metrics = {}
    
    def fastq_qc(self, fastq_file):
        """Quality control for FASTQ files."""
        
        total_reads = 0
        total_bases = 0
        quality_scores = []
        gc_content = []
        read_lengths = []
        
        with open(fastq_file, 'r') as handle:
            for record in SeqIO.parse(handle, "fastq"):
                total_reads += 1
                sequence = str(record.seq)
                total_bases += len(sequence)
                read_lengths.append(len(sequence))
                
                # GC content
                gc_count = sequence.count('G') + sequence.count('C')
                gc_content.append(gc_count / len(sequence) * 100)
                
                # Quality scores
                qualities = record.letter_annotations["phred_quality"]
                quality_scores.extend(qualities)
        
        self.qc_metrics['fastq'] = {
            'total_reads': total_reads,
            'total_bases': total_bases,
            'mean_read_length': sum(read_lengths) / len(read_lengths),
            'mean_quality': sum(quality_scores) / len(quality_scores),
            'mean_gc_content': sum(gc_content) / len(gc_content)
        }
        
        return self.qc_metrics['fastq']
    
    def alignment_qc(self, bam_file):
        """Quality control for alignment files."""
        
        samfile = pysam.AlignmentFile(bam_file, "rb")
        
        total_reads = 0
        mapped_reads = 0
        properly_paired = 0
        duplicates = 0
        mapping_qualities = []
        
        for read in samfile.fetch():
            total_reads += 1
            
            if not read.is_unmapped:
                mapped_reads += 1
                mapping_qualities.append(read.mapping_quality)
            
            if read.is_proper_pair:
                properly_paired += 1
                
            if read.is_duplicate:
                duplicates += 1
        
        samfile.close()
        
        self.qc_metrics['alignment'] = {
            'total_reads': total_reads,
            'mapped_reads': mapped_reads,
            'mapping_rate': mapped_reads / total_reads * 100,
            'properly_paired_rate': properly_paired / total_reads * 100,
            'duplicate_rate': duplicates / total_reads * 100,
            'mean_mapping_quality': sum(mapping_qualities) / len(mapping_qualities) if mapping_qualities else 0
        }
        
        return self.qc_metrics['alignment']
    
    def variant_qc(self, vcf_file):
        """Quality control for variant calls."""
        
        vcf = pysam.VariantFile(vcf_file)
        
        total_variants = 0
        snps = 0
        indels = 0
        quality_scores = []
        depth_values = []
        
        for record in vcf.fetch():
            total_variants += 1
            
            # Variant type
            if len(record.ref) == 1 and all(len(alt) == 1 for alt in record.alts):
                snps += 1
            else:
                indels += 1
            
            # Quality metrics
            if record.qual is not None:
                quality_scores.append(record.qual)
            
            # Depth information
            if 'DP' in record.info:
                depth_values.append(record.info['DP'])
        
        vcf.close()
        
        self.qc_metrics['variants'] = {
            'total_variants': total_variants,
            'snps': snps,
            'indels': indels,
            'snp_rate': snps / total_variants * 100,
            'mean_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'mean_depth': sum(depth_values) / len(depth_values) if depth_values else 0
        }
        
        return self.qc_metrics['variants']
    
    def generate_report(self, output_dir):
        """Generate comprehensive QC report."""
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # FASTQ metrics
        if 'fastq' in self.qc_metrics:
            fastq_data = self.qc_metrics['fastq']
            axes[0, 0].bar(['Total Reads', 'Total Bases', 'Mean Length', 'Mean Quality', 'GC Content'],
                          [fastq_data['total_reads'], fastq_data['total_bases'], 
                           fastq_data['mean_read_length'], fastq_data['mean_quality'],
                           fastq_data['mean_gc_content']])
            axes[0, 0].set_title('FASTQ Metrics')
        
        # Alignment metrics
        if 'alignment' in self.qc_metrics:
            align_data = self.qc_metrics['alignment']
            metrics = ['Mapping Rate', 'Properly Paired', 'Duplicate Rate']
            values = [align_data['mapping_rate'], align_data['properly_paired_rate'], 
                     align_data['duplicate_rate']]
            axes[0, 1].bar(metrics, values)
            axes[0, 1].set_title('Alignment Metrics (%)')
        
        # Variant metrics
        if 'variants' in self.qc_metrics:
            var_data = self.qc_metrics['variants']
            axes[1, 0].pie([var_data['snps'], var_data['indels']], 
                          labels=['SNPs', 'InDels'], autopct='%1.1f%%')
            axes[1, 0].set_title('Variant Types')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{self.sample_id}_qc_report.png", dpi=300, bbox_inches='tight')
        
        # Generate HTML report
        html_content = self._generate_html_report()
        with open(f"{output_dir}/{self.sample_id}_qc_report.html", 'w') as f:
            f.write(html_content)
    
    def _generate_html_report(self):
        """Generate HTML QC report."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QC Report - {self.sample_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Quality Control Report</h1>
            <h2>Sample: {self.sample_id}</h2>
            
            <div class="metric">
                <h3>FASTQ Metrics</h3>
                {self._metrics_table('fastq')}
            </div>
            
            <div class="metric">
                <h3>Alignment Metrics</h3>
                {self._metrics_table('alignment')}
            </div>
            
            <div class="metric">
                <h3>Variant Metrics</h3>
                {self._metrics_table('variants')}
            </div>
            
            <img src="{self.sample_id}_qc_report.png" alt="QC Plots" style="max-width: 100%;">
        </body>
        </html>
        """
        
        return html
    
    def _metrics_table(self, metric_type):
        """Generate HTML table for metrics."""
        
        if metric_type not in self.qc_metrics:
            return "<p>No data available</p>"
        
        metrics = self.qc_metrics[metric_type]
        table = "<table><tr><th>Metric</th><th>Value</th></tr>"
        
        for key, value in metrics.items():
            table += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.2f}</td></tr>"
        
        table += "</table>"
        return table

# Usage example
qc = GenomicsQC("sample_001")
qc.fastq_qc("sample_001_R1.fastq.gz")
qc.alignment_qc("sample_001.bam")
qc.variant_qc("sample_001.vcf.gz")
qc.generate_report("results/qc/")
```

## Analysis Workflows

### Multi-Omics Integration

```python
# Multi-omics data integration and analysis
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt

class MultiOmicsAnalysis:
    """Integrate and analyze multi-omics data."""
    
    def __init__(self):
        self.omics_data = {}
        self.integrated_data = None
        self.results = {}
    
    def load_omics_data(self, data_type, file_path, sample_col='sample_id'):
        """Load different types of omics data."""
        
        if data_type == 'genomics':
            # Load variant data
            df = pd.read_csv(file_path)
            # Convert to sample x variant matrix
            pivot_df = df.pivot_table(
                index=sample_col, 
                columns='variant_id', 
                values='genotype',
                fill_value=0
            )
            
        elif data_type == 'transcriptomics':
            # Load gene expression data
            df = pd.read_csv(file_path, index_col=0)
            pivot_df = df.T  # Samples as rows, genes as columns
            
        elif data_type == 'proteomics':
            # Load protein abundance data
            df = pd.read_csv(file_path)
            pivot_df = df.pivot_table(
                index=sample_col,
                columns='protein_id',
                values='abundance',
                fill_value=0
            )
            
        elif data_type == 'metabolomics':
            # Load metabolite data
            df = pd.read_csv(file_path)
            pivot_df = df.pivot_table(
                index=sample_col,
                columns='metabolite_id',
                values='concentration',
                fill_value=0
            )
        
        self.omics_data[data_type] = pivot_df
        print(f"Loaded {data_type} data: {pivot_df.shape}")
    
    def integrate_data(self, method='concatenation'):
        """Integrate multi-omics data."""
        
        if method == 'concatenation':
            # Simple concatenation of features
            data_list = []
            for omics_type, data in self.omics_data.items():
                # Standardize features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                scaled_df = pd.DataFrame(
                    scaled_data, 
                    index=data.index,
                    columns=[f"{omics_type}_{col}" for col in data.columns]
                )
                data_list.append(scaled_df)
            
            self.integrated_data = pd.concat(data_list, axis=1)
            
        elif method == 'pca_integration':
            # PCA-based integration
            integrated_features = []
            
            for omics_type, data in self.omics_data.items():
                # Apply PCA to each omics layer
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                
                pca = PCA(n_components=min(50, data.shape[1]))
                pca_features = pca.fit_transform(scaled_data)
                
                pca_df = pd.DataFrame(
                    pca_features,
                    index=data.index,
                    columns=[f"{omics_type}_PC{i+1}" for i in range(pca_features.shape[1])]
                )
                integrated_features.append(pca_df)
            
            self.integrated_data = pd.concat(integrated_features, axis=1)
        
        print(f"Integrated data shape: {self.integrated_data.shape}")
    
    def clustering_analysis(self, n_clusters=3):
        """Perform clustering on integrated data."""
        
        if self.integrated_data is None:
            raise ValueError("Data not integrated yet")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.integrated_data)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(self.integrated_data)
        
        # Store results
        self.results['clustering'] = {
            'clusters': clusters,
            'pca_coords': pca_coords,
            'explained_variance': pca.explained_variance_ratio_
        }
        
        # Plot results
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            pca_coords[:, 0], 
            pca_coords[:, 1], 
            c=clusters, 
            cmap='viridis'
        )
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Multi-Omics Clustering Analysis')
        plt.colorbar(scatter)
        plt.show()
        
        return clusters
    
    def pathway_analysis(self, gene_list, organism='human'):
        """Perform pathway enrichment analysis."""
        
        # Mock pathway analysis (would use KEGG, GO, etc.)
        pathways = {
            'Metabolic pathways': 0.001,
            'Cancer pathways': 0.005,
            'Immune response': 0.01,
            'Cell cycle': 0.02,
            'Apoptosis': 0.03
        }
        
        # Filter significant pathways
        significant_pathways = {k: v for k, v in pathways.items() if v < 0.05}
        
        self.results['pathways'] = significant_pathways
        
        # Plot pathway enrichment
        plt.figure(figsize=(10, 6))
        pathways_df = pd.DataFrame(
            list(significant_pathways.items()),
            columns=['Pathway', 'P-value']
        )
        pathways_df['-log10(p)'] = -np.log10(pathways_df['P-value'])
        
        plt.barh(pathways_df['Pathway'], pathways_df['-log10(p)'])
        plt.xlabel('-log10(P-value)')
        plt.title('Pathway Enrichment Analysis')
        plt.tight_layout()
        plt.show()
        
        return significant_pathways
    
    def network_analysis(self, correlation_threshold=0.7):
        """Build and analyze molecular interaction networks."""
        
        # Calculate correlation matrix
        corr_matrix = self.integrated_data.corr()
        
        # Create network graph
        G = nx.Graph()
        
        # Add edges based on correlation threshold
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > correlation_threshold:
                    G.add_edge(
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        weight=abs(corr_val)
                    )
        
        # Network analysis
        centrality = nx.degree_centrality(G)
        clustering = nx.clustering(G)
        
        self.results['network'] = {
            'graph': G,
            'centrality': centrality,
            'clustering': clustering
        }
        
        # Visualize network
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Node sizes based on centrality
        node_sizes = [centrality[node] * 1000 for node in G.nodes()]
        
        nx.draw(G, pos, 
                node_size=node_sizes,
                node_color='lightblue',
                edge_color='gray',
                alpha=0.7,
                with_labels=False)
        
        plt.title('Molecular Interaction Network')
        plt.show()
        
        return G

# Example usage
analysis = MultiOmicsAnalysis()

# Load different omics data
analysis.load_omics_data('transcriptomics', 'gene_expression.csv')
analysis.load_omics_data('proteomics', 'protein_abundance.csv')
analysis.load_omics_data('metabolomics', 'metabolite_data.csv')

# Integrate data
analysis.integrate_data(method='pca_integration')

# Perform analyses
clusters = analysis.clustering_analysis(n_clusters=3)
pathways = analysis.pathway_analysis(['BRCA1', 'TP53', 'EGFR'])
network = analysis.network_analysis(correlation_threshold=0.8)
```

## Related Resources

- [Cloud Infrastructure](../../deployment/cloud/)
- [Container Orchestration](../../deployment/kubernetes/)
- [ML Pipelines](../../machine-learning/)
- [Data Quality Tools](../../tools/data-quality/)

## Contributing

When adding new genomics content:
1. Follow FAIR data principles
2. Include proper citations
3. Document compute requirements
4. Provide example datasets
5. Add visualization examples 