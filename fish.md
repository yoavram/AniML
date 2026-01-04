# 🐟 Exercise: The Catch-and-Release Survey
**Role:** Field Ecologist
**Goal:** Automate species identification for a biodiversity survey.

**The Story:**
Your team is conducting a survey of a coastal reef. To minimize impact, you are using a **"Catch-and-Release"** protocol:
1.  Fish are caught in a net.
2.  You quickly measure their **Weight** (g) and dimensions (**Length, Height, Width** cm) on a portable field station.
3.  The fish are immediately released back into the water.

**The Problem:**
Identifying species in the field is slow and requires an expert taxonomist on the boat. We want to train an AI model that can identify the species based solely on these quick measurements, allowing volunteers to run future surveys.

**The Data:**
- Get it from [Mendeley Data](https://data.mendeley.com/datasets/bgsx9fjw4d/2): `Fish.csv`.
- **Target:** `Species` (Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt).
- **Features:**
  - `Weight`
  - `Length1`, `Length2`, `Length3` (Standard Length, Fork Length, Total Length)
  - `Height`, `Width`

**Your Tasks (Ask Gemini):**
1.  **Load** the data.
2.  **Sanity Check:** Field data often has errors. Filter out any rows where `Weight` is 0 or less (measurement error).
3.  **Explore:** Plot a **Boxplot of Weight vs Species**. Which species is the heaviest?
4.  **Train:** A Random Forest Classifier to predict `Species` from the measurements.
5.  **Visualize:** Plot the **Confusion Matrix**. Which species does the model struggle to distinguish?
