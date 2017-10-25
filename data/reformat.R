# Reformat given table matching reference images to customer images
# Using R instead of Python because laziness, basically

# read in csv
pills <- read.csv("groundTruthTable.csv")

# subset data table into "SF" and "SB" reference images
    # + corresponding customer quality images
pills_F <- pills[grep("_SF_", pills$ref_images), ]
pills_B <- pills[grep("_SB_", pills$ref_images), ]

# merge data tables
pills_refs <- merge(pills_F, pills_B, by="cons_images")

# final dataset is customer image -> both reference images
write.csv(pills_refs, "groundTruth_refs.csv")
