# Libraries

library(data.table)
library(Matrix)
library(recommenderlab)
library(Laurae)
library(fastdigest)
library(pbapply)
library(ggplot2)
library(R.utils)
library(stringi)

setwd("E:/")

my_script_is_using <- "E:/"
my_lgbm_is_at <- "C:/Compiled/LightGBM/windows/x64/Release/lightgbm.exe"
my_script_subbed <- basename(my_script_is_using)
threads <- 12
leaves <- 255
eta <- 0.01
min_sample <- 100
min_hess <- 10
subsample <- 0.7
colsample <- 0.7
sampling_freq <- ifelse(subsample == 1, 0, 1)


# Load from RDS

train <- readRDS("train.rds")
test <- readRDS("test.rds")
label <- readRDS("datasets/labels.rds")

to_keep <- which(colnames(train) %in% c("sameL0_next", "sameL1_next", "CATEGORICAL_Last_____1", "S24.311", 
                                        "gf1_1", "sameL3_next", "sameL6.0_V2_next", "gf0_1", "FOR30_Sum_S", 
                                        "CATEGORICAL_out_out_L3_S32_F3854_class2", "BAC60_Sum_S3", "DATE_S33max", 
                                        "DATE_L3kurt", "BAC30_Sum_S", "FOR60_Sum_S3", "BAC100_Sum", "S33", 
                                        "FOR100_log_lag_L3", "BAC100_log_lag_L3", "sameL0_prev", "S29", 
                                        "sameL3_prev", "L3_L3_Unique_Count", "FOR100_Sum", "Kurtosis", 
                                        "FOR165_Sum", "L3_S33_F3857", "L3_S30_F3754", "FOR100_log_lag", 
                                        "FOR165_log_lag_L3", "FOR165_log_lag", "S30", "L3_S33_F3859", 
                                        "L3_S30_F3759", "L3_S30_F3744", "BAC165_Sum", "BAC100_log_lag", 
                                        "L3_S30_F3749", "DATE_S34max", "L3_S33_F3865", "L3_S30_F3774", 
                                        "L3_S30_F3809", "BAC165_log_lag_L3", "L3_S30_F3804", "S1", "L3_S29_F3348", 
                                        "BAC165_log_lag", "L3_S29_F3373", "tdelta_devrel", "S32", "S0", 
                                        "L3_S29_F3351", "L3_S29_F3345", "L0_S1_F28", "d_29.0", "L3_S29_F3321", 
                                        "d_rel_d_29.0", "S24.308", "L3_S30_F3494", "L0_S0_F20_Mult_L0_S0_F20", 
                                        "L3_S29_F3324", "L3_S30_F3769", "Response_Minus1_Number1", "sameL6.0_V2_prev", 
                                        "L3_S30_F3554", "L3_S29_F3379", "L3_S29_F3427", "S29min_S32min", 
                                        "DATE_S32max", "S36", "L3_S29_F3342", "CATEGORICAL_Last_____2", 
                                        "Range", "Response_Minus1", "L3_S29_F3354", "L3_S30_F3704", "L3_S30_F3829", 
                                        "BAC60_log_lag_S34", "L0max_L3max", "L0_S1_F24", "L3_S30_F3609", 
                                        "L3_S29_F3479", "L3_S30_F3544", "BAC60_log_lag_S33", "L3_S36_F3920", 
                                        "L3max_L0min", "L3_S29_F3333", "FOR60_log_lag_S34", "DATE_S29max", 
                                        "L3_S30_F3574", "L3_S29_F3339", "S24.307", "S3", "L0_Max", "L0_S0_F20", 
                                        "L3_S29_F3327", "L3_S29_F3315", "L0_S9_F180", "L3_S29_F3336", 
                                        "L3_S29_F3382", "L3_Max", "L3_S30_F3794", "L3_S30_F3604", "L3_S29_F3376", 
                                        "sameL1_prev", "S2", "DATE_S37max", "L3_S30_F3534", "L0_S0_F18", 
                                        "L3_S30_F3519", "gf0_-1", "L0_S0_F16", "S35", "S33max_S29min", 
                                        "L3_S30_F3629", "L3_S30_F3514", "S5", "L3_S30_F3639", "S33min_S30min", 
                                        "S11", "L0_S0_F0", "L3_S29_F3330", "L3_S30_F3784", "L3_S30_F3524", 
                                        "FOR165_log_lag_L2", "L3_S29_F3318", "FOR60_log_lag_S33", "L3_S29_F3433", 
                                        "BAC165_log_lag_L2", "L0_S10_F244", "L3_S30_F3709", "L3_S30_F3509", 
                                        "L0_S0_F2", "L3_S30_F3819", "S7", "S10", "S6", "L3_S29_F3473", 
                                        "L3_S30_F3504", "L3_S30_F3624", "DATE_S36max", "S32_dest_33.0", 
                                        "L3_S30_F3764", "L2_Min", "DATE_S35max", "BAC165_log_lag_L0", 
                                        "S29min_S37min", "DATE_S30max", "L0_S2_F36", "L3_S29_F3436", 
                                        "BAC165_LAG2_Missing_Value_Count_ResponseMinus1", "L3_S30_F3799", 
                                        "L3_S29_F3430", "FOR60_log_lag_S30", "S4", "FOR165_log_lag_L1", 
                                        "BAC30_log_lag_S30", "L3_Min", "L0_S2_F44", "S12", "FOR100_log_lag_L2", 
                                        "BAC165_log_lag_L1", "L0_S7_F138", "BAC165_LAG2_Count_Mult_ResponseMinus1", 
                                        "FOR30_log_lag_S34", "L0_S5_F114", "Min", "S13", "FOR165_LAG_Missing_Value_Count_ResponseMinus1", 
                                        "Mean", "L0_S2_F60", "L0_S6_F132", "L0_S4_F104", "FOR100_LAG_Missing_Value_Count_ResponseMinus1", 
                                        "FOR165_LAG_Count_Mult_ResponseMinus1", "BAC30_log_lag_S33", 
                                        "BAC100_LAG_Missing_Value_Count_ResponseMinus1", "L0_S11_F322", 
                                        "FOR165_LAG1_Missing_Value_Count_ResponseMinus1", "L3_Range", 
                                        "FOR165_log_lag_L0", "L3_S30_F3579", "FOR165_LAG2_Missing_Value_Count_ResponseMinus1", 
                                        "S24.211", "BAC100_LAG3_Missing_Value_Count_ResponseMinus1", 
                                        "FOR100_LAG_Count_Mult_ResponseMinus1", "L0_S3_F100", "L0_S2_F64", 
                                        "BAC165_LAG_Missing_Value_Count_ResponseMinus1", "BAC30_log_lag_S34", 
                                        "FOR165_LAG3_Missing_Value_Count_ResponseMinus1", "BAC100_log_lag_L2", 
                                        "S29min_S34min", "L0_Mean", "L0_S5_F116", "L3_S29_F3357", "L0_S3_F72", 
                                        "S9", "FOR165_LAG1_Count_Mult_ResponseMinus1", "BAC100_LAG_Count_Mult_ResponseMinus1", 
                                        "BAC100_LAG2_Missing_Value_Count_ResponseMinus1", "sameL2_next", 
                                        "FOR30_log_lag_S30", "FOR165_LAG3_Count_Mult_ResponseMinus1", 
                                        "FOR165_LAG2_Count_Mult_ResponseMinus1", "BAC165_LAG_Count_Mult_ResponseMinus1", 
                                        "L0_S11_F310", "L3_S30_F3644", "BAC60_log_lag_S30", "L0_S3_F96", 
                                        "S30min_S37min", "BAC165_LAG1_Missing_Value_Count_ResponseMinus1", 
                                        "L3_S30_F3589", "L3_S30_F3569", "L3_S35_F3896", "L3_S30_F3669", 
                                        "DATE_S3_max", "FOR165_LAG0_Missing_Value_Count_ResponseMinus1", 
                                        "FOR100_LAG2_Missing_Value_Count_ResponseMinus1", "BAC165_LAG1_Count_Mult_ResponseMinus1", 
                                        "BAC100_log_lag_L1", "FOR100_LAG3_Missing_Value_Count_ResponseMinus1", 
                                        "L0_S6_F122", "L0_Min", "Unique_Count", "BAC100_LAG3_Count_Mult_ResponseMinus1", 
                                        "BAC165_LAG0_Missing_Value_Count_ResponseMinus1", "FOR165_LAG0_Count_Mult_ResponseMinus1", 
                                        "S8", "L3_S30_F3584", "FOR100_LAG2_Count_Mult_ResponseMinus1", 
                                        "L0_S4_F109", "BAC165_LAG3_Missing_Value_Count_ResponseMinus1", 
                                        "L3_S30_F3689", "BAC165_LAG0_Count_Mult_ResponseMinus1", "L3_S29_F3461", 
                                        "FOR100_LAG0_Missing_Value_Count_ResponseMinus1", "DATE_S6_max", 
                                        "BAC100_LAG2_Count_Mult_ResponseMinus1", "L0_S3_F80", "L3max_S30max", 
                                        "L3_S35_F3889", "L0_S0_F8", "FOR100_LAG1_Missing_Value_Count_ResponseMinus1", 
                                        "tdeltadevrel_block1", "FOR100_LAG3_Count_Mult_ResponseMinus1", 
                                        "L0_Range", "S36max_S29min", "BAC165_LAG3_Count_Mult_ResponseMinus1", 
                                        "L0_S0_F22", "L3_S30_F3684", "L0_S0_F6", "L0_S0_F10", "L0_S11_F326", 
                                        "L0_S7_F142", "sameL1.0_V2_prev", "DATE_S0_max", "FOR100_LAG0_Count_Mult_ResponseMinus1", 
                                        "BAC100_log_lag_L0", "S17", "L2_S26_F3062", "L0_S10_F254", "L0_S9_F170", 
                                        "FOR100_LAG1_Count_Mult_ResponseMinus1", "DATE_S5_max", "FOR100_log_lag_L0", 
                                        "DATE_L1kurt", "L3_S30_F3674", "L0_S9_F165", "BAC100_LAG1_Missing_Value_Count_ResponseMinus1", 
                                        "L2max_S37max", "L0_S0_F4", "L3max_S34min", "L0_S10_F229", "L0_S10_F259", 
                                        "L1_S24_F1723", "S26", "BAC100_LAG1_Count_Mult_ResponseMinus1", 
                                        "S33min_S37min", "DATE_S7_max", "tdeltadevrel_block1a", "BAC100_LAG0_Missing_Value_Count_ResponseMinus1", 
                                        "L2_S26_F3069", "S14", "L3_S29_F3476", "FOR100_log_lag_L1", "FOR30_log_lag_S33", 
                                        "L0_S13_F356", "L2_S26_F3106", "L0_S21_F497", "L0_S9_F155", "L0_S9_F185", 
                                        "L0_S10_F249", "sameL2_prev", "L0_S13_F354", "S19", "S27", "L0_S0_F14", 
                                        "L2_S27_F3210", "L3_S29_F3407", "S18", "L3_S30_F3564", "BAC100_LAG0_Count_Mult_ResponseMinus1", 
                                        "L0_S9_F195", "L0_S9_F160", "DATE_S2_max", "CATEGORICAL_Unique_Count", 
                                        "CATEGORICAL_Missing_Value_Count", "DATE_S4_max", "L3_S30_F3634", 
                                        "S37min_S34min", "L3_S30_F3664", "L0_S9_F190", "S32_dest_36.0", 
                                        "L0_S11_F302", "L0_S12_F348", "S30min_S35min", "S24.108", "L0_S12_F330", 
                                        "L0_S11_F306", "L1_S24_F1850", "S29max_S35max", "L0_S11_F294", 
                                        "L0_S0_F12", "L0_S12_F332", "S41", "L0_S12_F346", "L0_S10_F219", 
                                        "S34", "L1_S24_F1578", "S24.306", "gf1_-1", "L0_S11_F286", "L3_S30_F3679", 
                                        "L2_S27_F3162", "L0_S12_F350", "L1_S24_F1516", "L0_S11_F318", 
                                        "S16", "L0_S23_F643", "S45", "S15", "DATE_S10max", "L0_S10_F224", 
                                        "L2_S26_F3113", "L0_S11_F290", "FOR100_LAG_Sum_ResponseMinus1", 
                                        "S24.301", "L2_S26_F3121", "S38", "gf0_0", "FOR165_LAG3_Sum_ResponseMinus1", 
                                        "S25.109", "L0_S3_F84", "d_5.0", "S48", "DATE_S1_max", "L2_S27_F3129", 
                                        "S47", "L0_S11_F314_Mult_L0_S11_F314", "L0_S11_F314_Mult_L0_S0_F20", 
                                        "L2_S27_F3199", "d_rel_d_39.0", "DATE_S11max", "S24.303", "L3_Unique_Count", 
                                        "DATE_S18max", "L0_S17_F433", "S20", "S24.3", "L0_S11_F282", 
                                        "L0_S23_F667", "S35min_L1min", "d_rel_d_26.0", "d_27.0", "L1_S24_F1844", 
                                        "S23", "S13min_S33min", "L2_S26_F3047", "L3_S33_F3855", "L3_S29_F3360", 
                                        "L0_S11_F314", "S24.304", "DATE_S9_max", "L2_S26_F3040", "L2_S26_F3073", 
                                        "L3_S35_F3894", "L0_S2_F48", "DATE_S38max", "L3_S41_F4011", "L2_S27_F3155", 
                                        "L1_S24_F1773", "DATE_S8_max", "L1_Range", "S26max_S32max", "L0_S14_F386", 
                                        "L0_S9_F200", "L3_S32_F3850", "S24.104", "DATE_S16max", "S32_dest_35.0", 
                                        "d_36.0", "d_26.0", "DATE_S13max", "L1_S24_F1667", "L2_S26_F3036", 
                                        "L0_S9_F210", "S25.1", "S40", "L2_S26_F3117", "S21", "gf1_0", 
                                        "L0_Unique_Count", "L3max_S26max", "S33min_S35min", "L1_S24_F1662", 
                                        "L1_S24_F1514", "L0_S10_F264", "L1_Min", "S33min_S36min", "L0_S10_F239", 
                                        "L1_S24_F1520", "L2_Max", "L3_S41_F4008", "DATE_S26max", "S25.104", 
                                        "S24.305", "L0_S10_F234", "L2_S26_F3051", "L1_S24_F1788", "L1_Max", 
                                        "S22", "d_rel_d_32.0", "L0_S18_F439", "S32.0_comb_30.0-35.0", 
                                        "L3_S38_F3960", "L2_S27_F3166", "S24.205", "DATE_S14max", "L0_S22_F571", 
                                        "L0_S22_F591", "L0_S23_F639", "L2_S27_F3206", "L1_S24_F1518", 
                                        "L1_S24_F1831", "d_rel_d_27.0", "L1_S24_F1783", "L0_S12_F334", 
                                        "L1_S24_F1758", "L0_S12_F336", "d_rel_d_38.0", "L0_S15_F415", 
                                        "BAC100_LAG3_Sum_ResponseMinus1", "L1_S24_F1632", "L0_S14_F362", 
                                        "L0_S16_F421", "L2_S27_F3144", "L0_S21_F532", "L3_S36_F3918", 
                                        "L0_S14_F370", "L2_S27_F3214", "tdeltadevrel_block2", "DATE_S19max", 
                                        "L0_S19_F455", "L1_S24_F1463", "S50", "d_30.0", "L0_S22_F576", 
                                        "S24.109", "L3_S29_F3482", "DATE_S24max", "d_rel_d_36.0", "L0_S21_F482", 
                                        "sameL2.0_V2_prev", "S25.102", "d_33.0", "L1_S24_F1798", "L0_S12_F342", 
                                        "L1_S25_F2161", "L1_S24_F1829", "L1_S24_F1778", "L0_S15_F403", 
                                        "S24.309", "L0_S14_F390", "DATE_S17max", "L0_S15_F406", "DATE_S15max", 
                                        "L0_S16_F426", "L0_S23_F671", "L1_S24_F1652", "L3_L3_Missing_Value_Count", 
                                        "L0_S19_F459", "L1_S24_F1494", "L1_S24_F1544", "S24.31", "L0_S23_F655", 
                                        "DATE_S27max", "L1_S24_F1637", "L0_S15_F397", "S29.0_comb_-1.0-30.0", 
                                        "L1_S24_F1763", "L0_S12_F344", "L1_S24_F1700", "L1_S24_F1848", 
                                        "L0_S21_F522", "L1_S24_F1842", "L0_S15_F418", "S34min_S35min", 
                                        "L1_S24_F1824", "L1_S24_F1512", "L0_S18_F449", "L1_S24_F1573", 
                                        "S37", "d_rel_d_30.0", "S27min_S32min", "L0_S22_F606", "L3_S47_F4163", 
                                        "L2_S27_F3133", "L0_S10_F274", "L0_S3_F76", "sameL4.1_V2_prev", 
                                        "L1_S24_F1571", "S32min_S30min", "L1_S24_F1565", "S25.101", "L0_S2_F56", 
                                        "S25.106", "L0_S22_F556", "L1_S24_F1575", "L1_S24_F1567", "DATE_S21max", 
                                        "L1_S24_F1685", "d_14.0", "DATE_S22max", "L3_S48_F4198", "sameL3.3_V2_prev", 
                                        "S39", "L1_S24_F1581", "L3_S45_F4124", "L0_S2_F40", "L1_S24_F1812", 
                                        "d_4.0", "L0_S8_F144", "L0_S21_F492", "L0_S21_F487", "L1_S24_F1672", 
                                        "L0_S23_F631", "L0_S2_F32", "S26min_S24min", "L0_S14_F374", "S25.107", 
                                        "L0_S22_F586", "S32max_S37min", "L0_S12_F352", "L1_S25_F2116", 
                                        "L1_S24_F1846", "S25.105", "L0_S23_F627", "L0_S12_F338", "L0_S9_F175", 
                                        "L0_S21_F517", "L0_S17_F431", "S24.207", "S24.103", "L0_S21_F502", 
                                        "S33min_S34min", "S25.11", "d_38.0", "L1_S24_F1822", "L1_S25_F2136", 
                                        "L3_S47_F4153", "S49", "L0_S23_F623", "L3_S41_F4014", "L3_S41_F4026", 
                                        "d_35.0", "L0_S22_F596", "L3_S41_F4004", "L0_S23_F651", "L1_S25_F1973", 
                                        "L2_S27_F3140", "DATE_S24min", "L1_S25_F2126", "sameL1.0_V2_next", 
                                        "L0_S23_F659", "L1_S25_F1958", "S26min_S37min", "sameL7.0_V2_next", 
                                        "L0_S22_F551", "L1_S24_F1808", "d_7.0", "L0_S23_F619", "d_37.0", 
                                        "L0_S22_F601", "S24.2", "L3_S41_F4020", "S24.201", "L1_S24_F1818", 
                                        "DATE_S23max", "L0_S21_F507", "L1_S25_F2158", "L1_S25_F2176", 
                                        "sameL2.0_V2_next", "L0_S12_F340", "L3_S38_F3956", "L1_S24_F1748", 
                                        "d_34.0", "DATE_S25min", "L3_S40_F3984", "L0_S21_F477", "L1_S25_F2091", 
                                        "L0_S21_F512", "L0_S21_F527", "S25.211", "DATE_S20max", "L1_S25_F2131", 
                                        "d_39.0", "L3_S43_F4090", "L1_S24_F1647", "L0_S22_F561", "d_rel_d_33.0", 
                                        "L0_S3_F92", "BAC165_LAG_Sum_ResponseMinus1", "L0_S21_F537", 
                                        "L1_S24_F1728", "L0_S22_F546", "L3_S47_F4158", "L1_S25_F2106", 
                                        "L0_S23_F663", "FOR100_LAG3_Sum_ResponseMinus1", "S44", "L1_S24_F1743", 
                                        "L1_S24_F1820", "L3_S36_F3938", "L3_S48_F4196", "L3_S40_F3986", 
                                        "S37.0_comb_36.0--1.0", "L1_S24_F1539", "L1_S24_F1599", "L1_Unique_Count", 
                                        "S24.11", "DATE_S44max", "S43", "DATE_S12max", "S25.108", "L0_S3_F68", 
                                        "L1_S24_F1738", "L3_S47_F4138", "S24.206", "S24.202", "S24.112", 
                                        "L0_S21_F472", "L1_S24_F1753", "L1_S24_F1814", "L1_S25_F1892", 
                                        "FOR165_LAG_Sum_ResponseMinus1", "S24.209", "L3_S44_F4115", "S51", 
                                        "L3_S30_F3649", "L3_S41_F4006", "sameL7.0_V2_prev", "L1_S25_F2056", 
                                        "DATE_S25max", "L1_S24_F1816", "L1_S25_F2147", "L0_S7_F136", 
                                        "S37.0_comb_35.0--1.0", "L1_S25_F1938", "L0_S6_F118", "L3_S47_F4143", 
                                        "BAC165_LAG3_Sum_ResponseMinus1", "L3_S40_F3982", "d_rel_d_37.0", 
                                        "d_3.0", "S28", "d_6.0", "S25.225", "S24.101", "L1_S24_F1733", 
                                        "L1_S24_F1768", "L1_S25_F1858", "L1_S25_F1855", "d_rel_d_14.0", 
                                        "L1_S24_F1627", "S25.229", "S24.102", "L1_S24_F1498", "S31", 
                                        "d_2.0", "L1_S24_F1609", "L1_S24_F1690", "BAC165_LAG1_Sum_ResponseMinus1", 
                                        "L3_S40_F3980", "L1_S25_F2167", "L1_S25_F2155", "L0_S22_F611", 
                                        "S34.0_comb_33.0-36.0", "L1_S24_F1122", "sameL4.0_V2_prev", "d_rel_d_7.0", 
                                        "BAC100_LAG0_Sum_ResponseMinus1", "S37.0_comb_35.0-38.0", "FOR165_LAG2_Sum_ResponseMinus1", 
                                        "d_8.0", "S24.111", "FOR165_LAG0_Sum_ResponseMinus1", "BAC100_LAG_Sum_ResponseMinus1", 
                                        "S24.208", "BAC165_LAG2_Sum_ResponseMinus1", "S29.0_comb_10.0-30.0", 
                                        "L3_S41_F4016", "L3_S49_F4211", "S29.0_comb_9.0-30.0", "S34.0_comb_33.0-35.0", 
                                        "BAC165_LAG0_Sum_ResponseMinus1", "L1_S25_F2144", "L1_S24_F1718", 
                                        "L1_S24_F1713", "DATE_S41max", "S24.21", "d_rel_d_34.0", "L1_S25_F2173", 
                                        "L1_S24_F1490", "S24.107", "L1_S25_F2170", "S25.205", "S24.106", 
                                        "d_24.311", "L1_L1_Missing_Value_Count", "S25.226", "S25.223", 
                                        "S25.222", "L1_S24_F1202", "d_17.0", "L0_S20_F461", "S29.0_comb_21.0-30.0", 
                                        "FOR165_LAG1_Sum_ResponseMinus1", "DATE_S50max", "S27.0_comb_10.0-29.0", 
                                        "L1_S25_F1869", "d_rel_d_3.0", "S24.203", "FOR100_LAG2_Sum_ResponseMinus1", 
                                        "L1_S24_F1836", "S25.228", "d_16.0", "BAC100_LAG1_Sum_ResponseMinus1", 
                                        "L1_S24_F1838", "d_rel_d_6.0", "d_24.305", "d_19.0", "S25.224", 
                                        "d_18.0", "S35.0_comb_34.0-37.0", "FOR100_LAG1_Sum_ResponseMinus1", 
                                        "DATE_S48max", "FOR100_LAG0_Sum_ResponseMinus1", "d_rel_d_8.0", 
                                        "S25.204", "L1_S25_F1919", "d_9.0", "DATE_S47max", "L2_S26_F3077", 
                                        "S25.206", "L1_S25_F2152", "L1_S24_F1166", "DATE_S45min", "d_rel_d_2.0", 
                                        "d_24.307", "sameL3.2_V2_prev", "sameL3.3_V2_next", "L3_S44_F4118", 
                                        "L1_S25_F2051", "L1_S24_F1594", "L1_S25_F1978", "S32min_S10min", 
                                        "sameL3.1_V2_prev", "L1_S24_F1803", "L1_S24_F1642", "L1_S24_F1793", 
                                        "L1_S24_F1657", "d_32.0", "L1_S25_F2111", "L1_S24_F983", "DATE_S43max", 
                                        "L1_S25_F1992", "L1_S25_F2101", "L1_S24_F1622", "L3_S43_F4080", 
                                        "DATE_S40max", "DATE_S49max", "DATE_S28max", "CATEGORICAL_Max______3", 
                                        "d_13.0", "L1_S25_F1909", "d_10.0", "L3_S41_F4023", "L1_S25_F2031", 
                                        "S29.0_comb_23.0-30.0", "S37.0_comb_36.0-38.0", "L2_S26_F3125", 
                                        "L1_S24_F1810", "L3_S44_F4112", "L1_S24_F988", "d_15.0", "L1_S24_F1391", 
                                        "DATE_S51max", "S25.23", "S29.0_comb_11.0-30.0", "S32_dest_34.0", 
                                        "d_25.106", "L3_S44_F4121", "S30.0_comb_29.0-33.0", "L3_S30_F3499", 
                                        "L3_S41_F4000", "L1_S25_F2164", "L1_S25_F2016", "L0_S10_F269", 
                                        "d_1.0", "d_rel_d_15.0", "d_25.222", "d_11.0", "S25.203", "d_24.303", 
                                        "S25.207", "L0_S14_F366", "S25.227", "d_24.308", "d_25.109", 
                                        "S25.212", "Max", "L3_S47_F4148", "sameL4.4_V2_prev", "d_24.31", 
                                        "d_rel_d_10.0", "L1_S25_F2797", "L1_S24_F814", "d_rel_d_1.0", 
                                        "d_rel_d_11.0", "d_rel_d_48.0", "L3_S38_F3952", "d_rel_d_20.0", 
                                        "d_rel_d_9.0", "d_24.304", "S25.216", "d_21.0", "S25.221", "L1_S24_F1245", 
                                        "S25.214", "d_47.0", "S24.311_comb_nan-nan", "d_24.21", "CATEGORICAL_out_L3_S32_F3854_class1", 
                                        "S35.0_comb_32.0-37.0", "S32.0_comb_30.0-33.0", "S32.0_comb_30.0-36.0", 
                                        "L1_S24_F948", "L1_S25_F1900", "d_20.0", "L1_S24_F1041", "L3_S43_F4095", 
                                        "L3_S34_F3882", "S25.21", "L1_S24_F963", "L1_S24_F1366", "S36.0_comb_34.0-37.0", 
                                        "L0_S15_F400", "S29.0_comb_22.0-30.0", "S25.213", "d_24.306", 
                                        "S27.0_comb_9.0-29.0", "d_24.11", "L1_S25_F1968", "DATE_S31max", 
                                        "sameL4.2_V2_prev", "L1_S25_F2061", "L3_S31_F3842", "S32.0_comb_30.0-34.0", 
                                        "d_48.0", "BAC100_LAG2_Sum_ResponseMinus1", "sameL4.1_V2_next", 
                                        "L1_S24_F1834", "L3_S43_F4085", "d_24.309", "L3_S30_F3734", "S25.202", 
                                        "L0_S9_F205", "S25.208", "L1_S24_F810", "L2_S28_F3248", "S30.0_comb_29.0-32.0", 
                                        "S25.209", "L1_S24_F683", "d_rel_d_13.0", "d_rel_d_24.311", "d_rel_d_19.0", 
                                        "S24.311_comb_24.309-29.0", "L1_S25_F2041", "L1_S24_F1240", "L1_S25_F1953", 
                                        "d_rel_d_35.0", "d_rel_d_24.31", "L1_S24_F1180", "L1_S24_F1506", 
                                        "d_25.107", "L1_S24_F1482", "d_rel_d_16.0", "d_25.101", "sameL4.0_V2_next", 
                                        "L1_S25_F2071", "sameL3.1_V2_next", "S34.0_comb_32.0-35.0", "d_24.206", 
                                        "L3_S29_F3488", "d_24.209", "L1_S25_F2081", "d_rel_d_4.0", "L1_S25_F2086", 
                                        "L1_S25_F2007", "d_rel_d_25.107", "L1_S24_F1056", "L1_S24_F1197", 
                                        "L1_S24_F1006", "d_24.203", "d_50.0", "DATE_S39max", "d_24.301", 
                                        "d_25.11", "d_rel_d_24.305", "L1_S25_F1877", "d_rel_d_28.0", 
                                        "L1_S24_F1336", "S24.311_comb_24.309-26.0", "L1_S24_F733", "L1_S24_F1293", 
                                        "L1_S24_F1002", "d_44.0", "d_24.111", "d_25.105", "S36.0_comb_nan-nan", 
                                        "S25.201", "d_rel_d_25.105", "S32max_S25min", "d_rel_d_24.309", 
                                        "L3_S50_F4243", "S34.0_comb_nan-nan", "d_23.0", "S25.22", "S24.311_comb_24.31-27.0", 
                                        "d_rel_d_24.301", "S25.2", "d_24.211", "S29.0_comb_26.0-30.0", 
                                        "d_24.208", "d_22.0", "S27.0_comb_24.111-29.0", "d_25.104", "L2_S28_F3307", 
                                        "d_49.0", "L1_S24_F958", "sameL4.4_V2_next", "L1_S24_F691", "L1_S24_F1331", 
                                        "CATEGORICAL_Max______1", "L1_S24_F1321", "L1_S24_F829", "d_rel_d_24.206", 
                                        "d_25.206", "L2_L2_Unique_Count", "sameL3.2_V2_next", "d_rel_d_24.306", 
                                        "d_24.207", "d_rel_d_49.0", "d_rel_d_24.303", "S27.0_comb_25.11-29.0", 
                                        "S25.218", "L1_S24_F1000", "d_rel_d_17.0", "d_25.226", "S24.311_comb_24.309-27.0", 
                                        "L3_S37_F3950", "d_rel_d_24.203", "S25max_S32min", "L1_S25_F2837", 
                                        "d_24.108", "d_rel_d_18.0", "L1_S24_F1356", "S24.311_comb_24.31-26.0", 
                                        "d_rel_d_23.0", "sameL4.3_V2_next", "d_31.0", "d_24.101", "d_40.0", 
                                        "d_rel_d_21.0", "L1_S24_F1172", "d_rel_d_24.304", "d_rel_d_5.0", 
                                        "d_43.0", "d_rel_d_43.0", "L1_S24_F1303", "S27.0_comb_11.0-29.0", 
                                        "S27.0_comb_nan-nan", "d_rel_d_44.0", "d_45.0", "S27.0_comb_22.0-29.0", 
                                        "S25.219", "L1_S25_F1963", "d_25.102", "d_24.109", "S36.0_comb_32.0-37.0", 
                                        "d_rel_d_25.106", "d_rel_d_50.0", "S35.0_comb_nan-nan", "S27.0_comb_24.311-29.0", 
                                        "L1_S24_F700", "S29.0_comb_24.311-30.0", "d_24.102", "S25.217", 
                                        "S29.0_comb_25.11-30.0", "S29.0_comb_27.0-30.0", "d_rel_d_24.205", 
                                        "S24.311_comb_24.31-29.0", "d_rel_d_25.11", "d_25.228", "d_25.209", 
                                        "d_25.203", "d_rel_d_25.203", "d_24.105", "d_rel_d_24.105", "d_25.108", 
                                        "d_rel_d_25.101"))

train <- DTcolsample(train, to_keep)
test <- DTcolsample(test, to_keep)
gc()

train_mike <- fread("Mike/0 - Data Files/train_eng.csv")
test_mike <- fread("Mike/0 - Data Files/test_eng.csv")
train_mike[, Response := NULL]

to_keep <- which(colnames(train_mike) %in% c("sta_0m3", "sta_0m4", "sta_1m3", "sta_1m4", "sta_2m3", "sta_2m4", 
                                             "sta_3m3", "sta_3m4", "sta_4m3", "sta_4m4", "sta_5m3", "sta_5m4", 
                                             "sta_6m3", "sta_6m4", "sta_7m3", "sta_7m4", "sta_8m3", "sta_8m4", 
                                             "sta_9m3", "sta_9m4", "sta_10m3", "sta_10m4", "sta_11m3", "sta_11m4", 
                                             "sta_12m3", "sta_12m4", "sta_13m3", "sta_13m4", "sta_14m3", "sta_14m4", 
                                             "sta_15m3", "sta_15m4", "sta_16m3", "sta_16m4", "sta_17m3", "sta_17m4", 
                                             "sta_18m3", "sta_18m4", "sta_19m3", "sta_19m4", "sta_20m3", "sta_20m4", 
                                             "sta_21m3", "sta_21m4", "sta_22m3", "sta_22m4", "sta_23m3", "sta_23m4", 
                                             "sta_24m3", "sta_24m4", "sta_25m3", "sta_25m4", "sta_26m3", "sta_26m4", 
                                             "sta_27m3", "sta_27m4", "sta_28m3", "sta_28m4", "sta_29m3", "sta_29m4", 
                                             "sta_30m3", "sta_30m4", "sta_31m3", "sta_32m3", "sta_32m4", 
                                             "sta_33m3", "sta_33m4", "sta_34m3", "sta_34m4", "sta_35m3", "sta_35m4", 
                                             "sta_36m3", "sta_36m4", "sta_37m3", "sta_37m4", "sta_38m3", "sta_38m4", 
                                             "sta_39m3", "sta_39m4", "sta_40m3", "sta_40m4", "sta_41m3", "sta_41m4", 
                                             "sta_43m3", "sta_43m4", "sta_44m3", "sta_44m4", 
                                             "sta_45m3", "sta_45m4", "sta_47m3", "sta_47m4", 
                                             "sta_48m3", "sta_48m4", "sta_49m3", "sta_49m4", "sta_50m3", "sta_50m4", 
                                             "sta_51m3", "sta_51m4"))

train_mike <- DTcolsample(train_mike, to_keep)
test_mike <- DTcolsample(test_mike, to_keep)
gc()

train <- DTcbind(train, train_mike, low_mem = TRUE, collect = 8, silent = FALSE)
test <- DTcbind(test, test_mike, low_mem = TRUE, collect = 8, silent = FALSE)
rm(train_mike, test_mike)
gc()

# Sanity checking
spooky <- numeric(ncol(train))
for (i in 1:ncol(train)) {
  spooky[i] <- fastdigest(train[[i]])
  cat("Iteration ", i, " --- Uniques: ", length(unique(spooky[1:i])), " ---\n", sep = "")
}
# train <- DTcolsample(train, which(!duplicated(spooky)))
# test <- DTcolsample(test, which(!duplicated(spooky)))
# gc()





StratifiedCV <- function(Y, folds, seed) {
  folded <- list()
  folded1 <- list()
  folded2 <- list()
  set.seed(seed)
  temp_Y0 <- which(Y == 0)
  temp_Y1 <- which(Y == 1)
  for (i in 1:folds) {
    folded1[[i]] <- sample(temp_Y0, floor(length(temp_Y0) / ((folds + 1) - i)))
    temp_Y0 <- temp_Y0[!temp_Y0 %in% folded1[[i]]]
    folded2[[i]] <- sample(temp_Y1, floor(length(temp_Y1) / ((folds + 1) - i)))
    temp_Y1 <- temp_Y1[!temp_Y1 %in% folded2[[i]]]
    folded[[i]] <- c(folded1[[i]], folded2[[i]])
  }
  return(folded)
}

folds <- StratifiedCV(label, 5, 11111)


# Real cannon is here


temp_data <- lgbm.cv.prep(y_train = label,
                          x_train = train,
                          x_test = test,
                          data_has_label = FALSE,
                          NA_value = "nan",
                          workingdir = my_script_is_using,
                          folds = folds,
                          train_all = TRUE,
                          test_all = TRUE,
                          cv_all = TRUE)

temp_model <- lgbm.cv(y_train = label,
                      x_train = train,
                      x_test = test,
                      data_has_label = TRUE,
                      NA_value = "nan",
                      lgbm_path = my_lgbm_is_at,
                      workingdir = my_script_is_using,
                      files_exist = TRUE,
                      save_binary = FALSE,
                      validation = TRUE,
                      folds = folds,
                      predictions = TRUE,
                      importance = TRUE,
                      full_quiet = FALSE,
                      verbose = FALSE,
                      num_threads = threads,
                      application = "binary",
                      learning_rate = eta,
                      num_iterations = 5000,
                      early_stopping_rounds = 700,
                      num_leaves = leaves,
                      min_data_in_leaf = min_sample,
                      min_sum_hessian_in_leaf = min_hess,
                      max_bin = 255,
                      feature_fraction = colsample,
                      bagging_fraction = subsample,
                      bagging_freq = sampling_freq,
                      is_unbalance = FALSE,
                      metric = "auc",
                      is_training_metric = TRUE,
                      is_sparse = FALSE)

saveRDS(temp_model, file.path(my_script_is_using, "aaa_LightGBM_cv.rds"), compress = TRUE)

gc()

#temp_model <- readRDS(file.path(my_script_is_using, "aaa_LightGBM_cv.rds"))

best_iter <- 0
for (j in 1:5) {
  best_iter <- best_iter + 0.2 * temp_model$Models[[j]]$Best
}

cat("Going for ", best_iter, " rounds.  \n", sep = "")

best_model <- lgbm.train(y_train = label,
                         x_train = train,
                         x_test = test,
                         data_has_label = TRUE,
                         NA_value = "nan",
                         lgbm_path = my_lgbm_is_at,
                         workingdir = my_script_is_using,
                         files_exist = TRUE,
                         save_binary = FALSE,
                         predictions = TRUE,
                         importance = TRUE,
                         full_quiet = FALSE,
                         verbose = TRUE,
                         num_threads = threads,
                         application = "binary",
                         learning_rate = eta,
                         num_iterations = floor(best_iter * 1.1),
                         num_leaves = leaves,
                         min_data_in_leaf = min_sample,
                         min_sum_hessian_in_leaf = min_hess,
                         max_bin = 255,
                         feature_fraction = colsample,
                         bagging_fraction = subsample,
                         bagging_freq = sampling_freq,
                         is_unbalance = FALSE,
                         metric = "auc",
                         is_training_metric = TRUE,
                         is_sparse = FALSE)

saveRDS(best_model, file.path(my_script_is_using, "aaa_LightGBM_full.rds"), compress = TRUE)



mcc_fixed <- function(y_prob, y_true, prob) {
  
  positives <- as.logical(y_true) # label to boolean
  counter <- sum(positives) # get the amount of positive labels
  tp <- as.numeric(sum(y_prob[positives] > prob))
  fp <- as.numeric(sum(y_prob[!positives] > prob))
  tn <- as.numeric(length(y_true) - counter - fp) # avoid computing he opposite
  fn <- as.numeric(counter - tp) # avoid computing the opposite
  mcc <- (tp * tn - fp * fn) / (sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
  mcc <- ifelse(is.na(mcc), -1, mcc)
  return(mcc)
  
}

mcc_eval_print <- function(y_prob, y_true) {
  y_true <- y_true
  
  DT <- data.table(y_true = y_true, y_prob = y_prob, key = "y_prob")
  
  nump <- sum(y_true)
  numn <- length(y_true) - nump
  
  DT[, tn_v := as.numeric(cumsum(y_true == 0))]
  DT[, fp_v := cumsum(y_true == 1)]
  DT[, fn_v := numn - tn_v]
  DT[, tp_v := nump - fp_v]
  DT[, mcc_v := (tp_v * tn_v - fp_v * fn_v) / sqrt((tp_v + fp_v) * (tp_v + fn_v) * (tn_v + fp_v) * (tn_v + fn_v))]
  DT[, mcc_v := ifelse(!is.finite(mcc_v), 0, mcc_v)]
  gc(verbose = FALSE)
  
  return(max(DT[['mcc_v']]))
  
}

mcc_eval_pred <- function(y_prob, y_true) {
  y_true <- y_true
  
  DT <- data.table(y_true = y_true, y_prob = y_prob, key = "y_prob")
  
  nump <- sum(y_true)
  numn <- length(y_true) - nump
  
  DT[, tn_v := as.numeric(cumsum(y_true == 0))]
  DT[, fp_v := cumsum(y_true == 1)]
  DT[, fn_v := numn - tn_v]
  DT[, tp_v := nump - fp_v]
  DT[, mcc_v := (tp_v * tn_v - fp_v * fn_v) / sqrt((tp_v + fp_v) * (tp_v + fn_v) * (tn_v + fp_v) * (tn_v + fn_v))]
  DT[, mcc_v := ifelse(!is.finite(mcc_v), 0, mcc_v)]
  
  return(DT[['y_prob']][which.max(DT[['mcc_v']])])
  
}

FastROC <- function(y, x) {
  
  # y = actual
  # x = predicted
  x1 = as.numeric(x[y == 1])
  n1 = as.numeric(length(x1))
  x2 = as.numeric(x[y == 0])
  n2 = as.numeric(length(x2))
  r = rank(c(x1,x2))
  return((sum(r[1:n1]) - n1 * (n1 + 1) / 2) / (n1 * n2))
  
}




# Know what is inside


AnalysisFunc <- function(lgbm, label, folds, validationValues, predictedValuesCV, predictedValues) {
  # lgbm = your LightGBM cross-validated model (set as "NA" if it is not a LGBM model)
  # Label = your label
  # Folds = your fold list
  # validationValues = your validation values (out of fold predictions)
  # predictedValuesCV = your predicted values (on test data) from CV (set as "NA" if you don't have any)
  # predictedValues = your prediction values (on test data) on a model trained on all data (set as "NA" if you don't have any)
  
  if (length(lgbm) > 1) {
    
    # Print feature importance to file
    
    jpeg(filename = file.path(my_script_is_using, "importance_log_small.jpg"), width = 760, height = 894, units = "px", pointsize = 12)
    lgbm.fi.plot(temp_model, n_best = 50, no_log = FALSE, is.cv = TRUE, multipresence = TRUE, plot = TRUE)
    dev.off()
    jpeg(filename = file.path(my_script_is_using, "importance_nonlog_small.jpg"), width = 760, height = 894, units = "px", pointsize = 12)
    lgbm.fi.plot(temp_model, n_best = 50, no_log = TRUE, is.cv = TRUE, multipresence = TRUE, plot = TRUE)
    dev.off()
    
    jpeg(filename = file.path(my_script_is_using, "importance_log_big.jpg"), width = 760, height = 1788, units = "px", pointsize = 12)
    lgbm.fi.plot(temp_model, n_best = 100, no_log = FALSE, is.cv = TRUE, multipresence = TRUE, plot = TRUE)
    dev.off()
    jpeg(filename = file.path(my_script_is_using, "importance_nonlog_big.jpg"), width = 760, height = 1788, units = "px", pointsize = 12)
    lgbm.fi.plot(temp_model, n_best = 100, no_log = TRUE, is.cv = TRUE, multipresence = TRUE, plot = TRUE)
    dev.off()
    
  }
  
  # Setup tee
  sink(file = file.path(my_script_is_using, "diagnostics.txt"), append = FALSE, split = TRUE)
  cat("```r\n")
  
  if (length(lgbm) > 1) {
    
    # Get iteration information
    temp_iter <- numeric(5)
    best_iter <- 0
    for (j in 1:5) {
      temp_iter[j] <- lgbm$Models[[j]]$Best
      best_iter <- best_iter + (temp_iter[j] / length(folds))
      cat("Fold ", j, " converged after ", sprintf("%04d", temp_iter[j]), " iterations.\n", sep = "")
    }
    cat("Iterations: ", sprintf("%06.2f", mean(temp_iter)), " + ", sprintf("%06.3f", sd(temp_iter)), "\n\n\n", sep = "")
    
  }
  
  # Get AUC metric information
  temp_auc <- numeric(length(folds))
  best_auc <- 0
  for (j in 1:length(folds)) {
    temp_auc[j] <- FastROC(y = label[folds[[j]]], x = validationValues[folds[[j]]])
    best_auc <- best_auc + (temp_auc[j] / length(folds))
    cat("Fold ", j, ": AUC=", sprintf("%.07f", temp_auc[j]), "\n", sep = "")
  }
  cat("AUC: ", sprintf("%.07f", mean(temp_auc)), " + ", sprintf("%.07f", sd(temp_auc)), "\n", sep = "")
  cat("Average AUC using all data: ", sprintf("%.07f", FastROC(y = label, x = validationValues)), "\n\n\n", sep = "")
  
  
  # Get MCC metric information
  temp_mcc <- numeric(length(folds))
  temp_thresh <- numeric(length(folds))
  temp_positives <- numeric(length(folds))
  temp_detection <- numeric(length(folds))
  temp_true <- numeric(length(folds))
  temp_undetect <- numeric(length(folds))
  best_mcc <- 0
  for (j in 1:length(folds)) {
    
    temp_mcc[j] <- mcc_eval_print(y_prob = validationValues[folds[[j]]], y_true = label[folds[[j]]])
    temp_thresh[j] <- mcc_eval_pred(y_prob = validationValues[folds[[j]]], y_true = label[folds[[j]]])
    mini_preds <- validationValues[folds[[j]]] > temp_thresh[[j]]
    temp_positives[j] <- sum(mini_preds)
    temp_detection[j] <- 100 * temp_positives[j] / sum(label[folds[[j]]])
    temp_true[j] <- sum((mini_preds[mini_preds == TRUE] == label[folds[[j]]][mini_preds == TRUE]))
    temp_undetect[j] <- sum(label[folds[[j]]]) - temp_true[j]
    temp_true[j] <- 100 * temp_true[j] / sum(length(mini_preds[mini_preds == TRUE]))
    best_mcc <- best_mcc + (temp_mcc[j] / length(folds))
    cat("Fold ", j, ": MCC=", sprintf("%.07f", temp_mcc[j]), " (", sprintf("%04d", temp_positives[j]), " [", sprintf("%05.2f", temp_detection[j]), "%] positives), threshold=", sprintf("%.07f", temp_thresh[j]), " => True positives = ", sprintf("%06.3f", temp_true[j]), "%\n", sep = "")
    
  }
  cat("MCC: ", sprintf("%.07f", mean(temp_mcc)), " + ", sprintf("%.07f", sd(temp_mcc)), "\n", sep = "")
  cat("Threshold: ", sprintf("%.07f", mean(temp_thresh)), " + ", sprintf("%.07f", sd(temp_thresh)), "\n", sep = "")
  cat("Positives: ", sprintf("%06.2f", mean(temp_positives)), " + ", sprintf("%06.2f", sd(temp_positives)), "\n", sep = "")
  cat("Detection Rate %: ", sprintf("%06.3f", mean(temp_detection)), " + ", sprintf("%06.3f", sd(temp_detection)), "\n", sep = "")
  cat("True positives %: ", sprintf("%06.3f", mean(temp_true)), " + ", sprintf("%06.3f", sd(temp_true)), "\n", sep = "")
  cat("Undetected positives: ", sprintf("%07.2f", mean(temp_undetect)), " + ", sprintf("%07.2f", sd(temp_undetect)), "\n", sep = "")
  cat("Average MCC on all data (5 fold): ", sprintf("%.07f", mcc_fixed(y_prob = validationValues, y_true = label, prob = mean(temp_thresh))), ", threshold=", sprintf("%.07f", mean(temp_thresh)), "\n", sep = "")
  cat("Average MCC using all data: ", sprintf("%.07f", mcc_eval_print(y_prob = validationValues, y_true = label)), ", threshold=", sprintf("%.07f", mcc_eval_pred(y_prob = validationValues, y_true = label)), "\n\n\n", sep = "")
  
  
  if (length(predictedValuesCV) > 1) {
    
    # Create overfitted submission from all data
    best_mcc1 <- mcc_eval_pred(y_prob = validationValues, y_true = label)
    submission0_ov <- fread("datasets/sample_submission.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
    submission0_ov$Response <- as.numeric(predictedValuesCV > best_mcc1)
    best_count1 <- sum(submission0_ov$Response == 1)
    cat("Submission overfitted threshold on all MCC positives: ", best_count1, "\n\n", sep = "")
    write.csv(submission0_ov, file = file.path(my_script_is_using, paste(my_script_subbed, "_val_", sprintf("%.06f", best_mcc1), "_", best_count1, ".csv", sep = "")), row.names = FALSE)
    
    # Create CV submission from validation
    best_mcc2 <- mean(temp_thresh)
    submission0 <- fread("datasets/sample_submission.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
    submission0$Response <- as.numeric(predictedValuesCV > best_mcc2)
    best_count2 <- sum(submission0$Response == 1)
    cat("Submission average validated threshold on all MCC positives: ", best_count2, "\n\n", sep = "")
    write.csv(submission0, file = file.path(my_script_is_using, paste(my_script_subbed, "_val_", sprintf("%.06f", best_mcc2), "_", best_count2, ".csv", sep = "")), row.names = FALSE)
    
    # Create average of the two previous submissions
    best_mcc3 <- (best_mcc1 + best_mcc2) / 2
    submission0_ex <- fread("datasets/sample_submission.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
    submission0_ex$Response <- as.numeric(predictedValuesCV > best_mcc3)
    best_count3 <- sum(submission0_ex$Response == 1)
    cat("Submission average of overfit+validated threshold positives: ", best_count3, "\n\n", sep = "")
    write.csv(submission0_ex, file = file.path(my_script_is_using, paste(my_script_subbed, "_val_", sprintf("%.06f", best_mcc3), "_", best_count3, ".csv", sep = "")), row.names = FALSE)
    
    # Create files for stacker
    write.csv(validationValues, file = file.path(my_script_is_using, "aaa_stacker_preds_train_headerY.csv"), row.names = FALSE)
    write.csv(predictedValuesCV, file = file.path(my_script_is_using, "aaa_stacker_preds_test_headerY.csv"), row.names = FALSE)
    
  }
  
  
  if (length(predictedValues) > 1) {
    
    # Create overfitted submission from all data using full trained model
    best_mcc1_all <- best_mcc1
    submission0_ov_all <- fread("datasets/sample_submission.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
    submission0_ov_all$Response <- as.numeric(predictedValues > best_mcc1_all)
    best_count1_all <- sum(submission0_ov_all$Response == 1)
    cat("Submission with all data overfitted threshold on all MCC positives: ", best_count1_all, ". Threshold=", best_mcc1_all, "\n\n", sep = "")
    write.csv(submission0_ov_all, file = file.path(my_script_is_using, paste(my_script_subbed, "_all_", sprintf("%.06f", best_mcc1_all), "_", best_count1_all, ".csv", sep = "")), row.names = FALSE)
    
    # Create CV submission from validation using full trained model
    best_mcc2_all <- best_mcc2
    submission0_all <- fread("datasets/sample_submission.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
    submission0_all$Response <- as.numeric(predictedValues > best_mcc2_all)
    best_count2_all <- sum(submission0_all$Response == 1)
    cat("Submission with all data average validated threshold on all MCC positives: ", best_count2_all, ". Threshold=", best_mcc2_all, "\n\n", sep = "")
    write.csv(submission0_all, file = file.path(my_script_is_using, paste(my_script_subbed, "_all_", sprintf("%.06f", best_mcc2_all), "_", best_count2_all, ".csv", sep = "")), row.names = FALSE)
    
    # Create average of the two previous submissions using full trained model
    best_mcc3_all <- best_mcc3
    submission0_ex_all <- fread("datasets/sample_submission.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
    submission0_ex_all$Response <- as.numeric(predictedValues > best_mcc3_all)
    best_count3_all <- sum(submission0_ex_all$Response == 1)
    cat("Submission with all data average of overfit+validated threshold positives: ", best_count3_all, ". Threshold=", best_mcc3_all, "\n\n", sep = "")
    write.csv(submission0_ex_all, file = file.path(my_script_is_using, paste(my_script_subbed, "_all_", sprintf("%.06f", best_mcc3_all), "_", best_count3_all, ".csv", sep = "")), row.names = FALSE)
    
    
    
    mini_preds <- predictedValues
    mini_preds <- mini_preds[order(mini_preds, decreasing = TRUE)]
    
    # Create overfitted submission from all data using full trained model using respective positive count
    best_mcc1_all_val <- mini_preds[best_count1 + 1]
    submission0_ov_all_val <- fread("datasets/sample_submission.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
    submission0_ov_all_val$Response <- as.numeric(predictedValues > best_mcc1_all_val)
    best_count1_all_val <- sum(submission0_ov_all_val$Response == 1)
    cat("Submission with all data by taking the amount of positives of overfitted threshold on all MCC positives: ", best_count1_all_val, ". Threshold=", best_mcc1_all_val, "\n\n", sep = "")
    write.csv(submission0_ov_all_val, file = file.path(my_script_is_using, paste(my_script_subbed, "_all_val_", sprintf("%.06f", best_mcc1_all_val), "_", best_count1_all_val, ".csv", sep = "")), row.names = FALSE)
    
    # Create CV submission from validation using full trained model using respective positive count
    best_mcc2_all_val <- mini_preds[best_count2 + 1]
    submission0_all_val <- fread("datasets/sample_submission.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
    submission0_all_val$Response <- as.numeric(predictedValues > best_mcc2_all_val)
    best_count2_all_val <- sum(submission0_all_val$Response == 1)
    cat("Submission with all data by taking the amount of positives of average validated threshold on all MCC positives: ", best_count2_all_val, ". Threshold=", best_mcc2_all_val, "\n\n", sep = "")
    write.csv(submission0_all_val, file = file.path(my_script_is_using, paste(my_script_subbed, "_all_val_", sprintf("%.06f", best_mcc2_all_val), "_", best_count2_all_val, ".csv", sep = "")), row.names = FALSE)
    
    # Create average of the two previous submissions using full trained model using respective positive count
    best_mcc3_all_val <- mini_preds[best_count3 + 1]
    submission0_ex_all_val <- fread("datasets/sample_submission.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
    submission0_ex_all_val$Response <- as.numeric(predictedValues > best_mcc3_all_val)
    best_count3_all_val <- sum(submission0_ex_all_val$Response == 1)
    cat("Submission with all data by taking the amount of positives of average of overfit+validated threshold positives: ", best_count3_all_val, ". Threshold=", best_mcc3_all_val, "\n\n", sep = "")
    write.csv(submission0_ex_all_val, file = file.path(my_script_is_using, paste(my_script_subbed, "_all_val_", sprintf("%.06f", best_mcc3_all_val), "_", best_count3_all_val, ".csv", sep = "")), row.names = FALSE)
    
    # Create submissions using full trained model using total validated positive count
    best_mcc_extra <- mini_preds[sum(temp_positives) + 1]
    submission0_extra <- fread("datasets/sample_submission.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
    submission0_extra$Response <- as.numeric(predictedValues > best_mcc_extra)
    best_count_extra <- sum(submission0_extra$Response == 1)
    cat("Submission with all data by taking the sum of positives of validated positives: ", best_count_extra, ". Threshold=", best_mcc_extra, "\n\n", sep = "")
    write.csv(submission0_extra, file = file.path(my_script_is_using, paste(my_script_subbed, "_extra_", sprintf("%.06f", best_mcc_extra), "_", best_count_extra, ".csv", sep = "")), row.names = FALSE)
    
  }
  
  
  if (length(lgbm) > 1) {
    
    # Do best feature listing
    cat("Cross-validated used features list (all used features to copy & paste):\n\n")
    mini_model <- copy(lgbm$FeatureImp)
    dput(mini_model$Feature)
    cat("\n\nCross-validated multipresence used features list (all used features to copy & paste):\n\n")
    mini_model <- mini_model[Freq == length(folds), ]
    dput(mini_model$Feature)
    
  }
  
  cat("```")
  sink()
  
}


AnalysisFunc(lgbm = temp_model,
             label = label,
             folds = folds,
             validationValues = temp_model$Validation[[1]],
             predictedValuesCV = temp_model$Testing[[1]],
             predictedValues = best_model$Testing)
