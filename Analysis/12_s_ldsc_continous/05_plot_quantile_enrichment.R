#!/usr/bin/env Rscript
#
# Plot quantile enrichment results
#
# This script creates visualizations of the quantile-based enrichment analysis
# showing how heritability is distributed across quantiles of continuous annotations
#

library(ggplot2)
library(dplyr)
library(tidyr)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  cat("\nUsage: Rscript 05_plot_quantile_enrichment.R [summary_file] [output_prefix]\n\n")
  cat("Arguments:\n")
  cat("  summary_file   : Path to summary_enrichment.txt file\n")
  cat("  output_prefix  : Prefix for output plot files\n\n")
  cat("If no arguments provided, will use default paths:\n")
  cat("  summary_file   = quantile_results/summary_enrichment.txt\n")
  cat("  output_prefix  = quantile_results/enrichment_plot\n\n")

  # Use defaults
  script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
  if (length(script_dir) == 0) script_dir <- "."

  summary_file <- file.path(script_dir, "quantile_results", "summary_enrichment.txt")
  output_prefix <- file.path(script_dir, "quantile_results", "enrichment_plot")
} else if (length(args) == 2) {
  summary_file <- args[1]
  output_prefix <- args[2]
} else {
  stop("Error: Provide either 0 or 2 arguments")
}

# Check if file exists
if (!file.exists(summary_file)) {
  stop(paste("Error: Summary file not found:", summary_file))
}

# Read data
cat("Reading data from:", summary_file, "\n")
data <- read.table(summary_file, header = TRUE, sep = "\t")

cat("Loaded", nrow(data), "rows for", length(unique(data$Track)), "track(s)\n")
cat("Tracks:", paste(unique(data$Track), collapse = ", "), "\n\n")

# Create output directory if needed
output_dir <- dirname(output_prefix)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Theme for plots
theme_publication <- theme_bw(base_size = 12) +
  theme(
    panel.grid.major = element_line(size = 0.5, color = "grey90"),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95", color = "grey80"),
    legend.position = "bottom",
    legend.key.size = unit(0.8, "cm")
  )

# Plot 1: Proportion of heritability by quantile
cat("Creating plot 1: Proportion of heritability by quantile...\n")
p1 <- ggplot(data, aes(x = Quantile, y = prop_h2g, color = Track, group = Track)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = prop_h2g - prop_h2g_se, ymax = prop_h2g + prop_h2g_se),
                width = 0.2, alpha = 0.6) +
  labs(
    title = "Proportion of Heritability Across Quantiles",
    x = "Quantile of Continuous Annotation",
    y = "Proportion of h²g",
    color = "Track"
  ) +
  scale_x_continuous(breaks = unique(data$Quantile)) +
  theme_publication

ggsave(paste0(output_prefix, "_prop_h2g.pdf"), p1, width = 10, height = 6)
ggsave(paste0(output_prefix, "_prop_h2g.png"), p1, width = 10, height = 6, dpi = 300)

# Plot 2: Enrichment by quantile
cat("Creating plot 2: Enrichment by quantile...\n")
p2 <- ggplot(data, aes(x = Quantile, y = enr, color = Track, group = Track)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = enr - enr_se, ymax = enr + enr_se),
                width = 0.2, alpha = 0.6) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
  labs(
    title = "Enrichment Across Quantiles",
    subtitle = "Dashed line indicates no enrichment (enrichment = 1)",
    x = "Quantile of Continuous Annotation",
    y = "Enrichment",
    color = "Track"
  ) +
  scale_x_continuous(breaks = unique(data$Quantile)) +
  theme_publication

ggsave(paste0(output_prefix, "_enrichment.pdf"), p2, width = 10, height = 6)
ggsave(paste0(output_prefix, "_enrichment.png"), p2, width = 10, height = 6, dpi = 300)

# Plot 3: Heritability (h2g) by quantile
cat("Creating plot 3: Heritability by quantile...\n")
p3 <- ggplot(data, aes(x = Quantile, y = h2g, color = Track, group = Track)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = h2g - h2g_se, ymax = h2g + h2g_se),
                width = 0.2, alpha = 0.6) +
  labs(
    title = "Heritability Across Quantiles",
    x = "Quantile of Continuous Annotation",
    y = "h²g",
    color = "Track"
  ) +
  scale_x_continuous(breaks = unique(data$Quantile)) +
  theme_publication

ggsave(paste0(output_prefix, "_h2g.pdf"), p3, width = 10, height = 6)
ggsave(paste0(output_prefix, "_h2g.png"), p3, width = 10, height = 6, dpi = 300)

# Plot 4: Significance heatmap
cat("Creating plot 4: Enrichment significance heatmap...\n")

# Add significance levels
data$significance <- cut(data$enr_pval,
                        breaks = c(0, 0.001, 0.01, 0.05, 1),
                        labels = c("p < 0.001", "p < 0.01", "p < 0.05", "n.s."),
                        include.lowest = TRUE)

p4 <- ggplot(data, aes(x = factor(Quantile), y = Track, fill = enr)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = ifelse(enr_pval < 0.05,
                               sprintf("%.2f*", enr),
                               sprintf("%.2f", enr))),
            size = 3) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red",
    midpoint = 1, limits = c(min(0.5, min(data$enr)), max(2, max(data$enr))),
    name = "Enrichment"
  ) +
  labs(
    title = "Enrichment Heatmap",
    subtitle = "* indicates p < 0.05",
    x = "Quantile",
    y = "Track"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    panel.grid = element_blank()
  )

ggsave(paste0(output_prefix, "_heatmap.pdf"), p4, width = 10, height = max(6, length(unique(data$Track)) * 0.5))
ggsave(paste0(output_prefix, "_heatmap.png"), p4, width = 10, height = max(6, length(unique(data$Track)) * 0.5), dpi = 300)

# Summary statistics
cat("\n" + rep("=", 70) + "\n")
cat("Summary Statistics\n")
cat(rep("=", 70) + "\n\n")

for (track in unique(data$Track)) {
  track_data <- data[data$Track == track, ]

  cat("Track:", track, "\n")
  cat("------\n")
  cat(sprintf("  Total h²g: %.4f ± %.4f\n", sum(track_data$h2g),
              sqrt(sum(track_data$h2g_se^2))))

  # Find quantiles with significant enrichment
  sig_quantiles <- track_data[track_data$enr_pval < 0.05, ]
  if (nrow(sig_quantiles) > 0) {
    cat("  Significantly enriched quantiles (p < 0.05):\n")
    for (i in 1:nrow(sig_quantiles)) {
      cat(sprintf("    Q%d: Enrichment = %.3f ± %.3f, p = %.2e\n",
                  sig_quantiles$Quantile[i],
                  sig_quantiles$enr[i],
                  sig_quantiles$enr_se[i],
                  sig_quantiles$enr_pval[i]))
    }
  } else {
    cat("  No significantly enriched quantiles (p < 0.05)\n")
  }
  cat("\n")
}

cat("All plots saved with prefix:", output_prefix, "\n")
cat("  - Proportion of h²g: ", output_prefix, "_prop_h2g.pdf/png\n", sep = "")
cat("  - Enrichment: ", output_prefix, "_enrichment.pdf/png\n", sep = "")
cat("  - h²g: ", output_prefix, "_h2g.pdf/png\n", sep = "")
cat("  - Heatmap: ", output_prefix, "_heatmap.pdf/png\n", sep = "")
cat("\nDone!\n")
