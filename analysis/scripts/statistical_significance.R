# install.packages("lme4")
# install.packages("lmerTest")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("brms")
# install.packages("emmeans")
# install.packages("progress")
# install.packages("tidyverse")

# Load the required libraries
library(lme4) # For fitting GLMMs
library(lmerTest) # Provides p-values for fixed effects in lme4 models
library(dplyr) # For data manipulation
library(ggplot2) # For plotting
library(brms)
library(emmeans) # for follow-up pairwise comparisons
library(progress) # for progress bars
# library(tidyverse) # for data manipulation


# Load the data
data_humans <- read.csv("experiments/gardenpath_10_24/results/human_results/sampled_results.csv")
data_llms <- read.csv("experiments/gardenpath_10_24/results/llm_results/all_results.csv")
data_llms <- data_llms %>% filter(model == "google/gemma-2-27b-it") # Remove the first model, which is the null model

head(data)

# Convert variables to factors
data_humans$SentenceType <- factor(data_humans$SentenceType, levels = c("nonGP", "GP")) # Reference level is 'nonGP'
data_humans$ManipulationType <- factor(data_humans$ManipulationType, levels = c("prob", "improb", "reflexive"))
data_humans$quest_type <- factor(data_humans$quest_type)

# Convert variables to factors
data_llms$SentenceType <- factor(data_llms$SentenceType, levels = c("nonGP", "GP")) # Reference level is 'nonGP'
data_llms$ManipulationType <- factor(data_llms$ManipulationType, levels = c("prob", "improb", "reflexive"))
data_llms$quest_type <- factor(data_llms$quest_type)



# descriptive statistics:
with(data_humans, tapply(corr, list(SentenceType, ManipulationType, quest_type), mean))
with(data_llms, tapply(correct, list(SentenceType, ManipulationType, quest_type), mean))
# trans_factor: corr ~ SentenceType X Transitivity + (1 + SentenceType | set_id)

h1_data_hum_1 <- data_humans %>%
    filter(quest_type == "GP_question", SentenceType == "nonGP", ManipulationType %in% c("prob", "reflexive"))

h1_model1_hum_1 <- glmer(
    corr ~ isReflexive + (1 + isReflexive | set_id),
    data = h1_data_hum_1,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

# We first check if passing from GP to nonGP is significant for the reflexive condition

h1_data_hum <- data_llms %>%
    filter(ManipulationType == "reflexive", quest_type == "GP_question")

h1_model2_hum <- glmer(
    correct ~ SentenceType + (1 | set_id),
    data = h1_data_hum,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

model_summary <- summary(h1_model2_hum)

capture.output(
    {
        cat("Summary for model: 1\n") # Add a header for clarity
        print(model_summary)
        cat("\n====================\n\n") # Add some space between summaries
    },
    file = "/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/gemma-2-27b-it.txt"
)


h2_data_hum <- data_llms %>%
    filter(ManipulationType == "prob", quest_type == "GP_question")

h2_model_hum <- glmer(
    correct ~ SentenceType + (1 + SentenceType | set_id),
    data = h2_data_hum,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

model_summary <- summary(h2_model_hum)

capture.output(
    {
        cat("Summary for model: 2\n") # Add a header for clarity
        print(model_summary)
        cat("\n====================\n\n") # Add some space between summaries
    },
    file = "/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/gemma-2-27b-it.txt",
    append = TRUE
)



h3_data_hum <- data_llms %>%
    filter(ManipulationType == "improb", quest_type == "GP_question")

h3_model_hum <- glmer(
    correct ~ SentenceType + (1 + SentenceType | set_id),
    data = h3_data_hum,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

model_summary <- summary(h3_model_hum)

capture.output(
    {
        cat("Summary for model: 3\n") # Add a header for clarity
        print(model_summary)
        cat("\n====================\n\n") # Add some space between summaries
    },
    file = "/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/gemma-2-27b-it.txt",
    append = TRUE
)


h4_data_hum <- data_llms %>%
    filter(ManipulationType %in% c("prob", "improb"), quest_type == "GP_question")

# sum-code the variables, so that we don't have to use ANOVA
h4_data_hum$SentenceType <- factor(h4_data_hum$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
(contrasts(h4_data_hum$SentenceType) <- contrasts(h4_data_hum$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))

h4_data_hum$ManipulationType <- factor(h4_data_hum$ManipulationType, levels = c("improb", "prob")) # Ref level = improb
(contrasts(h4_data_hum$ManipulationType) <- contrasts(h4_data_hum$ManipulationType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))


h4_model2_hum <- glmer(
    correct ~ SentenceType * ManipulationType + (1 + ManipulationType | set_id),
    data = h4_data_hum,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

model_summary <- summary(h4_model2_hum)

capture.output(
    {
        cat("Summary for model: 4\n") # Add a header for clarity
        print(model_summary)
        cat("\n====================\n\n") # Add some space between summaries
    },
    file = "/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/gemma-2-27b-it.txt",
    append = TRUE
)

# Aya: Instead of running the models you had above for h2 and h3, we look at pairwise comparisons as follow-ups to
# this model. Both turn out significant, but only the improb remains significant after correction
# for two comparisons.

emmeans_summary <- emmeans(h4_model2_hum, pairwise ~ SentenceType | ManipulationType)

capture.output(
    {
        cat("Summary for model: 4 bis\n") # Add a header for clarity
        print(emmeans_summary)
        cat("\n====================\n\n") # Add some space between summaries
    },
    file = "/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/gemma-2-27b-it.txt",
    append = TRUE
)


# We check if there is any interaction between the transitivity factor, the SentenceType and the ManipulationType

h5_data_hum <- data_llms %>%
    filter(ManipulationType %in% c("prob", "improb"), quest_type == "GP_question") %>%
    filter(trans_factor != -1) # Exclude trans_factor == -1

h5_data_hum$ManipulationType <- factor(h5_data_hum$ManipulationType, levels = c("improb", "prob")) # Ref level = improb
(contrasts(h5_data_hum$ManipulationType) <- contrasts(h5_data_hum$ManipulationType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))

h5_data_hum$SentenceType <- factor(h5_data_hum$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
(contrasts(h5_data_hum$SentenceType) <- contrasts(h5_data_hum$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))

h5_data_hum$trans_factor_c <- scale(h5_data_hum$trans_factor, center = TRUE, scale = FALSE)

# observation: the interaction between sentence type and transition factor is significant
# Singular fit warning, simplifying the model:


h5_model2_hum <- glmer(
    correct ~ ManipulationType * SentenceType * trans_factor_c +
        (1 + ManipulationType | set_id),
    data = h5_data_hum, family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa")
)

model_summary <- summary(h5_model2_hum)

capture.output(
    {
        cat("Summary for model: 5\n") # Add a header for clarity
        print(model_summary)
        cat("\n====================\n\n") # Add some space between summaries
    },
    file = "/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/gemma-2-27b-it.txt",
    append = TRUE
)


h6_data_hum <- data_llms %>%
    filter(ManipulationType == "prob", quest_type == "GP_question") %>%
    filter(trans_factor != -1) # Exclude trans_factor == -1

h6_data_hum$trans_factor_c <- scale(h6_data_hum$trans_factor, center = TRUE, scale = FALSE)

h6_data_hum$SentenceType <- factor(h6_data_hum$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
(contrasts(h6_data_hum$SentenceType) <- contrasts(h6_data_hum$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))


h6_model_hum <- glmer(
    correct ~ SentenceType * trans_factor_c +
        (1 + SentenceType | set_id),
    data = h6_data_hum, family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa")
)

# observation: the interaction between sentence type and transition factor is significant
# Aya: Again, what's the direction of this interaction?
model_summary <- summary(h6_model_hum)

capture.output(
    {
        cat("Summary for model: 6\n") # Add a header for clarity
        print(model_summary)
        cat("\n====================\n\n") # Add some space between summaries
    },
    file = "/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/gemma-2-27b-it.txt",
    append = TRUE
)


h7_data_hum <- data_llms %>%
    filter(ManipulationType == "improb", quest_type == "GP_question") %>%
    filter(trans_factor != -1) # Exclude trans_factor == -1

h7_data_hum$trans_factor_c <- scale(h7_data_hum$trans_factor, center = TRUE, scale = FALSE)

h7_data_hum$SentenceType <- factor(h7_data_hum$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
(contrasts(h7_data_hum$SentenceType) <- contrasts(h7_data_hum$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))


h7_model_hum <- glmer(
    correct ~ SentenceType * trans_factor_c +
        (1 + SentenceType | set_id),
    data = h7_data_hum, family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa")
)

# observation: the sentence type influence is significant
model_summary <- summary(h7_model_hum)

capture.output(
    {
        cat("Summary for model: 7\n") # Add a header for clarity
        print(model_summary)
        cat("\n====================\n\n") # Add some space between summaries
    },
    file = "/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/gemma-2-27b-it.txt",
    append = TRUE
)

# # We check if passing from GP to nonGP is significant for the prob condition for simple question

h8_data_hum <- data_llms %>%
    filter(ManipulationType == "prob", quest_type == "simple_question")

h8_model_hum <- glmer(
    correct ~ SentenceType + (1 | set_id),
    data = h8_data_hum,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)
# observation: the sentence type influence is significant
model_summary <- summary(h8_model_hum)

capture.output(
    {
        cat("Summary for model: 8\n") # Add a header for clarity
        print(model_summary)
        cat("\n====================\n\n") # Add some space between summaries
    },
    file = "/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/gemma-2-27b-it.txt",
    append = TRUE
)

# We check if there is any interaction between the differences and the SentenceType  and manipulation type
h9_data_hum <- data_llms %>%
    filter(ManipulationType %in% c("prob", "improb"), quest_type == "GP_question") %>%
    filter(plausibility != -1) # Exclude trans_factor == -1

h9_data_hum$plausibility_c <- scale(h9_data_hum$plausibility, center = TRUE, scale = FALSE)

h9_model_hum <- glmer(
    correct ~ ManipulationType * SentenceType * plausibility_c +
        (1 + ManipulationType | set_id),
    data = h9_data_hum, family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa")
)

# observation: the interaction between manipulation type and differences in prob vs. improb is significant
model_summary <- summary(h9_model_hum)
anova_summary <- anova(h9_model_hum)

capture.output(
    {
        cat("Summary for model: 9\n") # Add a header for clarity
        print(model_summary)
        cat("\nAnova for model: 9\n") # Add a header for clarity
        print(anova_summary)
        cat("\n====================\n\n") # Add some space between summaries
    },
    file = "/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/gemma-2-27b-it.txt",
    append = TRUE
)

# unique_models <- list("Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-14B", "vicuna-13b-v1.5", "Llama-3.1-70B")
# unique_models <- unique(data_llms$short_model)

# # Create a progress bar object
# pb <- progress_bar$new(
#     format = "[:bar] :current/:total (:percent) :elapsed",
#     total = length(unique_models),
#     clear = FALSE, width = 60
# )

# for (model_name in unique_models) {
#     output_file_name <- paste0("/Users/samuelamouyal/PycharmProjects/reading_comprehension_research/experiments/gardenpath_10_24/analysis/significance_results/llms/", model_name, ".txt")

#     curr_data_model <- data_llms %>%
#         filter(short_model == model_name)

#     h1_data_mod <- curr_data_model %>%
#         filter(ManipulationType == "reflexive", quest_type == "GP_question")

#     h1_model2_mod <- try(glmer(
#         correct ~ SentenceType + (1 | set_id),
#         data = h1_data_mod,
#         family = binomial()(link = "log"),
#         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
#     ))

#     # Check if the model fitting resulted in an error
#     if (inherits(h1_model2_mod, "try-error")) {
#         capture.output(
#             {
#                 cat("Model 1 fitting failed with an error.\n")
#             },
#             file = output_file_name
#         )
#         #  Optionally, handle the error further, log the error, or try different strategies
#     } else {
#         model_summary <- summary(h1_model2_mod)
#         capture.output(
#             {
#                 cat("Summary for model: 1\n") # Add a header for clarity
#                 print(model_summary)
#                 cat("\n====================\n\n") # Add some space between summaries
#             },
#             file = output_file_name
#         )
#     }

#     h2_data_mod <- curr_data_model %>%
#         filter(ManipulationType == "prob", quest_type == "GP_question")

#     h2_model_mod <- try(glmer(
#         correct ~ SentenceType + (1 | set_id),
#         data = h2_data_mod,
#         family = binomial(link = "logit"),
#         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
#     ))

#     # Check if the model fitting resulted in an error
#     if (inherits(h2_model_mod, "try-error")) {
#         capture.output(
#             {
#                 cat("Model 2 fitting failed with an error.\n")
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#         #  Optionally, handle the error further, log the error, or try different strategies
#     } else {
#         model_summary <- summary(h2_model_mod)
#         capture.output(
#             {
#                 cat("Summary for model: 2\n") # Add a header for clarity
#                 print(model_summary)
#                 cat("\n====================\n\n") # Add some space between summaries
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#     }

#     h3_data_mod <- curr_data_model %>%
#         filter(ManipulationType == "improb", quest_type == "GP_question")

#     h3_model_mod <- try(glmer(
#         correct ~ SentenceType + (1 | set_id),
#         data = h3_data_mod,
#         family = binomial(link = "logit"),
#         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
#     ))

#     # Check if the model fitting resulted in an error
#     if (inherits(h3_model_mod, "try-error")) {
#         capture.output(
#             {
#                 cat("Model 3 fitting failed with an error.\n")
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#         #  Optionally, handle the error further, log the error, or try different strategies
#     } else {
#         model_summary <- summary(h3_model_mod)
#         capture.output(
#             {
#                 cat("Summary for model: 3\n") # Add a header for clarity
#                 print(model_summary)
#                 cat("\n====================\n\n") # Add some space between summaries
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#     }


#     h4_data_mod <- curr_data_model %>%
#         filter(ManipulationType %in% c("prob", "improb"), quest_type == "GP_question")

#     # sum-code the variables, so that we don't have to use ANOVA
#     h4_data_mod$SentenceType <- factor(h4_data_mod$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
#     (contrasts(h4_data_mod$SentenceType) <- contrasts(h4_data_mod$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))

#     h4_data_mod$ManipulationType <- factor(h4_data_mod$ManipulationType, levels = c("improb", "prob")) # Ref level = improb
#     (contrasts(h4_data_mod$ManipulationType) <- contrasts(h4_data_mod$ManipulationType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))


#     h4_model2_mod <- try(glmer(
#         correct ~ SentenceType * ManipulationType + (1 | set_id),
#         data = h4_data_mod,
#         family = binomial(link = "logit"),
#         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
#     ))

#     # Check if the model fitting resulted in an error
#     if (inherits(h4_model2_mod, "try-error")) {
#         capture.output(
#             {
#                 cat("Model 4 fitting failed with an error.\n")
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#         #  Optionally, handle the error further, log the error, or try different strategies
#     } else {
#         model_summary <- summary(h4_model2_mod)
#         capture.output(
#             {
#                 cat("Summary for model: 4\n") # Add a header for clarity
#                 print(model_summary)
#                 cat("\n====================\n\n") # Add some space between summaries
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#     }

#     # Aya: Instead of running the models you had above for h2 and h3, we look at pairwise comparisons as follow-ups to
#     # this model. Both turn out significant, but only the improb remains significant after correction
#     # for two comparisons.

#     emmeans_summary <- try(emmeans(h4_model2_mod, pairwise ~ SentenceType | ManipulationType))

#     if (inherits(emmeans_summary, "try-error")) {
#         capture.output(
#             {
#                 cat("Model 4 emmeans failed with an error.\n")
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#         #  Optionally, handle the error further, log the error, or try different strategies
#     } else {
#         model_summary <- summary(h4_model2_mod)
#         capture.output(
#             {
#                 cat("Summary for model: 4 bis\n") # Add a header for clarity
#                 print(emmeans_summary)
#                 cat("\n====================\n\n") # Add some space between summaries
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#     }

#     # We check if there is any interaction between the transitivity factor, the SentenceType and the ManipulationType

#     h5_data_mod <- curr_data_model %>%
#         filter(ManipulationType %in% c("prob", "improb"), quest_type == "GP_question") %>%
#         filter(trans_factor != -1) # Exclude trans_factor == -1

#     h5_data_mod$ManipulationType <- factor(h5_data_mod$ManipulationType, levels = c("improb", "prob")) # Ref level = improb
#     (contrasts(h5_data_mod$ManipulationType) <- contrasts(h5_data_mod$ManipulationType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))

#     h5_data_mod$SentenceType <- factor(h5_data_mod$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
#     (contrasts(h5_data_mod$SentenceType) <- contrasts(h5_data_mod$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))

#     h5_data_mod$trans_factor_c <- scale(h5_data_mod$trans_factor, center = TRUE, scale = FALSE)

#     # observation: the interaction between sentence type and transition factor is significant
#     # Singular fit warning, simplifying the model:


#     h5_model2_mod <- try(glmer(
#         correct ~ ManipulationType * SentenceType * trans_factor_c +
#             (1 | set_id),
#         data = h5_data_mod, family = binomial(link = "logit"),
#         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
#     ))

#     if (inherits(h5_model2_mod, "try-error")) {
#         capture.output(
#             {
#                 cat("Model 5 fitting failed with an error.\n")
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#         #  Optionally, handle the error further, log the error, or try different strategies
#     } else {
#         model_summary <- summary(h5_model2_mod)
#         capture.output(
#             {
#                 cat("Summary for model: 5\n") # Add a header for clarity
#                 print(model_summary)
#                 cat("\n====================\n\n") # Add some space between summaries
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#     }


#     h6_data_mod <- curr_data_model %>%
#         filter(ManipulationType == "prob", quest_type == "GP_question") %>%
#         filter(trans_factor != -1) # Exclude trans_factor == -1

#     h6_data_mod$trans_factor_c <- scale(h6_data_mod$trans_factor, center = TRUE, scale = FALSE)

#     h6_data_mod$SentenceType <- factor(h6_data_mod$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
#     (contrasts(h6_data_mod$SentenceType) <- contrasts(h6_data_mod$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))


#     h6_model_mod <- try(glmer(
#         correct ~ SentenceType * trans_factor_c +
#             (1 | set_id),
#         data = h6_data_mod, family = binomial(link = "logit"),
#         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
#     ))

#     if (inherits(h6_model_mod, "try-error")) {
#         capture.output(
#             {
#                 cat("Model 6 fitting failed with an error.\n")
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#         #  Optionally, handle the error further, log the error, or try different strategies
#     } else {
#         model_summary <- summary(h6_model_mod)
#         capture.output(
#             {
#                 cat("Summary for model: 6\n") # Add a header for clarity
#                 print(model_summary)
#                 cat("\n====================\n\n") # Add some space between summaries
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#     }


#     h7_data_mod <- curr_data_model %>%
#         filter(ManipulationType == "improb", quest_type == "GP_question") %>%
#         filter(trans_factor != -1) # Exclude trans_factor == -1

#     h7_data_mod$trans_factor_c <- scale(h7_data_mod$trans_factor, center = TRUE, scale = FALSE)

#     h7_data_mod$SentenceType <- factor(h7_data_mod$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
#     (contrasts(h7_data_mod$SentenceType) <- contrasts(h7_data_mod$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))


#     h7_model_mod <- try(glmer(
#         correct ~ SentenceType * trans_factor_c +
#             (1 | set_id),
#         data = h7_data_mod, family = binomial(link = "logit"),
#         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
#     ))

#     if (inherits(h7_model_mod, "try-error")) {
#         capture.output(
#             {
#                 cat("Model 7 fitting failed with an error.\n")
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#         #  Optionally, handle the error further, log the error, or try different strategies
#     } else {
#         model_summary <- summary(h7_model_mod)
#         capture.output(
#             {
#                 cat("Summary for model: 7\n") # Add a header for clarity
#                 print(model_summary)
#                 cat("\n====================\n\n") # Add some space between summaries
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#     }

#     # We check if passing from GP to nonGP is significant for the prob condition for simple question

#     h8_data_mod <- curr_data_model %>%
#         filter(ManipulationType == "prob", quest_type == "simple_question")

#     h8_model_mod <- try(glmer(
#         correct ~ SentenceType + (1 | set_id),
#         data = h8_data_mod,
#         family = binomial(link = "logit"),
#         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
#     ))

#     if (inherits(h8_model_mod, "try-error")) {
#         capture.output(
#             {
#                 cat("Model 8 fitting failed with an error.\n")
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#         #  Optionally, handle the error further, log the error, or try different strategies
#     } else {
#         model_summary <- summary(h8_model_mod)
#         capture.output(
#             {
#                 cat("Summary for model: 8\n") # Add a header for clarity
#                 print(model_summary)
#                 cat("\n====================\n\n") # Add some space between summaries
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#     }

#     # We check if there is any interaction between the differences and the SentenceType  and manipulation type
#     h9_data_mod <- curr_data_model %>%
#         filter(ManipulationType %in% c("prob", "improb"), quest_type == "GP_question") %>%
#         filter(plausibility != -1) # Exclude trans_factor == -1

#     h9_data_mod$plausibility_c <- scale(h9_data_mod$plausibility, center = TRUE, scale = FALSE)

#     h9_model_mod <- try(glmer(
#         correct ~ ManipulationType * SentenceType * plausibility_c +
#             (1 | set_id),
#         data = h9_data_mod, family = binomial(link = "logit"),
#         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
#     ))

#     if (inherits(h9_model_mod, "try-error")) {
#         capture.output(
#             {
#                 cat("Model 9 fitting failed with an error.\n")
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#         #  Optionally, handle the error further, log the error, or try different strategies
#     } else {
#         model_summary <- summary(h9_model_mod)
#         capture.output(
#             {
#                 cat("Summary for model: 9\n") # Add a header for clarity
#                 print(model_summary)
#                 cat("\nAnova for model: 9\n") # Add a header for clarity
#                 print(anova_summary)
#                 cat("\n====================\n\n") # Add some space between summaries
#             },
#             file = output_file_name,
#             append = TRUE
#         )
#     }
#     pb$tick()
# }

# h1_model <- glmer(
#     corr ~ SentenceType + (1 + SentenceType | set_id),
#     data = h1_data,
#     family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
# )


# # observation: significant effect
# summary(h1_model)

# the model above gave a "singular fit" warning"  - probably due to overfitting. To simplify it, I
# took out the random slope of SentenceType (which explained little variance). The new model
# below now converges, with similar significance.

# h1_model2 <- glmer(
#     corr ~ SentenceType + (1 | set_id),
#     data = h1_data,
#     family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
# )

# summary(h1_model2)


# # We first check if passing from GP to nonGP is significant for the prob condition

# h2_data <- data %>%
#     filter(ManipulationType == "prob", quest_type == "GP_question")

# h2_model <- glmer(
#     corr ~ SentenceType + (1 + SentenceType | set_id),
#     data = h2_data,
#     family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
# )

# # observation: not significant - but not far! p = .07
# summary(h2_model)

# # We first check if passing from GP to nonGP is significant for the improb condition

# h3_data <- data %>%
#     filter(ManipulationType == "improb", quest_type == "GP_question")

# h3_model <- glmer(
#     corr ~ SentenceType + (1 + SentenceType | set_id),
#     data = h3_data,
#     family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
# )

# # observation: significant effect
# summary(h3_model)

# # We check if passing from GP to nonGP is significant for the other conditions and we see if there is some interaction
# # between SentenceType and ManipulationType



# h4_data <- data %>%
#     filter(ManipulationType %in% c("prob", "improb"), quest_type == "GP_question")

# # sum-code the variables, so that we don't have to use ANOVA
# h4_data$SentenceType <- factor(h4_data$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
# (contrasts(h4_data$SentenceType) <- contrasts(h4_data$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))

# h4_data$ManipulationType <- factor(h4_data$ManipulationType, levels = c("improb", "prob")) # Ref level = improb
# (contrasts(h4_data$ManipulationType) <- contrasts(h4_data$ManipulationType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))

# h4_model <- glmer(
#     corr ~ SentenceType * ManipulationType + (1 + SentenceType * ManipulationType | set_id),
#     data = h4_data,
#     family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
# )

# summary(h4_model)
# # observation: if we allow random slope of SentenceType * ManipulationType based on set_id, the model converges
# # but the SentenceType influence is not significant. However, if we do not allow random slope, the SentenceType influence
# # is significant
# # Aya: it converges but with a singular fit warning. I'm simplifying the model - taking out the
# # random effect which explains the least variance (the interaction - like you did). Yes, now it converges
# # with sentence type significant.

# h4_model2 <- glmer(
#     corr ~ SentenceType * ManipulationType + (1 + SentenceType + ManipulationType | set_id),
#     data = h4_data,
#     family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
# )

# summary(h4_model2)

# # Aya: Instead of running the models you had above for h2 and h3, we look at pairwise comparisons as follow-ups to
# # this model. Both turn out significant, but only the improb remains significant after correction
# # for two comparisons.

# emmeans(h4_model2, pairwise ~ SentenceType | ManipulationType)



# # We check if there is any interaction between the transitivity factor, the SentenceType and the ManipulationType

# h5_data <- data %>%
#     filter(ManipulationType %in% c("prob", "improb"), quest_type == "GP_question") %>%
#     filter(trans_factor != -1) # Exclude trans_factor == -1

# h5_data$ManipulationType <- factor(h5_data$ManipulationType, levels = c("improb", "prob")) # Ref level = improb
# (contrasts(h5_data$ManipulationType) <- contrasts(h5_data$ManipulationType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))

# h5_data$SentenceType <- factor(h5_data$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
# (contrasts(h5_data$SentenceType) <- contrasts(h5_data$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))

# h5_data$trans_factor_c <- scale(h5_data$trans_factor, center = TRUE, scale = FALSE)

# h5_model <- glmer(
#     corr ~ ManipulationType * SentenceType * trans_factor_c +
#         (1 + ManipulationType * SentenceType | set_id),
#     data = h5_data, family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa")
# )
# summary(h5_model)

# # observation: the interaction between sentence type and transition factor is significant
# # Singular fit warning, simplifying the model:


# h5_model2 <- glmer(
#     corr ~ ManipulationType * SentenceType * trans_factor_c +
#         (1 + ManipulationType + SentenceType | set_id),
#     data = h5_data, family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa")
# )

# summary(h5_model2)

# # Results: Main effect of manipulation type (probable less accurate than imrpobable);
# # Main effect of sentence type (GP less accurate than nonGP)
# # Interaction of transitivity and sentence type: **what's the direction of this interaction**?
# # i.e. does transitivity have *more* effect in GP? in non-GP? in what direction?


# # We check if there is any interaction between the transitivity factor, the SentenceType for prob

# h6_data <- data %>%
#     filter(ManipulationType == "prob", quest_type == "GP_question") %>%
#     filter(trans_factor != -1) # Exclude trans_factor == -1

# h6_data$trans_factor_c <- scale(h6_data$trans_factor, center = TRUE, scale = FALSE)

# h6_data$SentenceType <- factor(h6_data$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
# (contrasts(h6_data$SentenceType) <- contrasts(h6_data$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))


# h6_model <- glmer(
#     corr ~ SentenceType * trans_factor_c +
#         (1 + SentenceType | set_id),
#     data = h6_data, family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa")
# )

# # observation: the interaction between sentence type and transition factor is significant
# # Aya: Again, what's the direction of this interaction?
# summary(h6_model)


# # We check if there is any interaction between the transitivity factor, the SentenceType for improb

# h7_data <- data %>%
#     filter(ManipulationType == "improb", quest_type == "GP_question") %>%
#     filter(trans_factor != -1) # Exclude trans_factor == -1

# h7_data$trans_factor_c <- scale(h7_data$trans_factor, center = TRUE, scale = FALSE)

# h7_data$SentenceType <- factor(h7_data$SentenceType, levels = c("GP", "nonGP")) # Ref level = GP
# (contrasts(h7_data$SentenceType) <- contrasts(h7_data$SentenceType) - matrix(rep(1 / 2, 2 * 1), nrow = 2))


# h7_model <- glmer(
#     corr ~ SentenceType * trans_factor_c +
#         (1 + SentenceType | set_id),
#     data = h7_data, family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa")
# )

# # observation: the sentence type influence is significant
# summary(h7_model)


# # We check if passing from GP to nonGP is significant for the prob condition for simple question

# h8_data <- data %>%
#     filter(ManipulationType == "prob", quest_type == "simple_question")

# h8_model <- glmer(
#     corr ~ SentenceType + (1 + SentenceType | set_id),
#     data = h8_data,
#     family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
# )

# # observation: not significant
# summary(h8_model)


# # We check if there is any interaction between the differences and the SentenceType  and manipulation type
# h9_data <- data %>%
#     filter(ManipulationType %in% c("prob", "improb"), quest_type == "GP_question") %>%
#     filter(diffs != -1) # Exclude trans_factor == -1

# h9_data$diffs_c <- scale(h9_data$diffs, center = TRUE, scale = FALSE)

# h9_model <- glmer(
#     corr ~ ManipulationType * SentenceType * diffs_c +
#         (1 + ManipulationType * SentenceType | set_id),
#     data = h9_data, family = binomial(link = "logit"),
#     control = glmerControl(optimizer = "bobyqa")
# )

# # observation: the interaction between manipulation type and differences in prob vs. improb is significant
# summary(h9_model)
# anova(h9_model)
