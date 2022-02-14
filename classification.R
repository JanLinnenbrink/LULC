# load packages -----------------------------------------------------------

rm(list=ls())

library(SpaDES)
library(sf)
library(raster)
library(caret)
library(CAST)
library(dplyr)
library(tmap)
library(mapview)
library(readxl)
library(rasterVis)
library(doParallel)
library(RStoolbox)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))


# load data ---------------------------------------------------------------

preds <- stack("preds.grd")
pr_rs <- readRDS("Trainingsdaten_Guetersloh_Jan.RDS")

pred_names <- c("B02","B03","B04","B08","B05","B06","B07","B11",
                "B12","NDVI","NDVI_5x5_sd","NDVI_3x3_sd")

marburg <- stack("Sentinel_Marburg.grd")


# explore relationships between predictors and response --------------------------------

trainIDs <- createDataPartition(pr_rs$Label,p=0.1,list = FALSE)
trainDat <- pr_rs[trainIDs,]
trainDat <- merge(trainDat, luc_tab, by="Label")

luc_i <- c("mixed forest" ,"fields planted", "village", "lake")

col <- c("lightgreen","blue4","green4", "grey")
my_settings <- list(superpose.symbol=list(col=col,fill= col))

png("feature_plot.png", width=15, height=15, units="cm",res = 800)

featurePlot(x=trainDat[trainDat$Label_en %in% luc_i,
                       c("B04","B08","NDVI","NDVI_3x3_sd")],
            y=factor(trainDat[trainDat$Label_en %in% luc_i,]$Label_en),
            plot="pairs",
            auto.key = list(columns = 2),
            par.settings=my_settings)

dev.off()

# spatial ffs -------------------------------------------------------------

trainDat <- pr_rs

# cv-method
indices <- CreateSpacetimeFolds(trainDat, spacevar = "ID", k=4)
ctrl <- trainControl(method="cv",index = indices$index,savePredictions=TRUE)

# train model
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

set.seed(10)
ffs_model <- ffs(predictors = trainDat[, !names(trainDat) %in% c("ID","Label","Region")], 
                 response = trainDat$Label,
                 ntree = 50,
                 tuneLength = 3,
                 method = "rf",
                 metric = "Kappa",
                 importance = TRUE,
                 trControl = ctrl)

stopCluster(cl)

saveRDS(ffs_model, "model.RDS")

# get test metrics
cm <- caret::confusionMatrix(ffs_model)$table
library(xtable)
acr <- c("Fp","Fu","V","W","G","I","Mf","Sea","St")

colnames(cm) <- acr
rownames(cm) <- acr

print(xtable(cm, type = "latex"), file = "confusion_matrix.tex")


cmdf <- as.data.frame(cm)
# here we also have the rounded percentage values
cm_p <- as.data.frame(prop.table(cm))
cmdf$Perc <- round(cm_p$Freq*100,2)

cm_d_p <-  ggplot(data = cmdf, aes(x = Prediction , y =  Reference, fill = Freq))+
  geom_tile() +
  geom_text(aes(label = paste("",Freq,",",Perc,"%")), color = 'red', size = 4) +
  theme_light() +
  guides(fill=FALSE) 

# get details about the model
ffs_model$finalModel
ffs_model$selectedvars

png("ffs_res.png", width=14, height=8, units="cm",res = 800)
plot_ffs(ffs_model = ffs_model)
dev.off()


# get variable importance
vi <- caret::varImp(ffs_model)
colnames(vi$importance) <- c("fields unplanted", "fields planted", "village", 
                             "watercourse", "grassland", "industry", "mixed forest",
                             "sea", "traffic road")

png("vimp.png", width=15, height=12, units="cm",res = 800)
plot(vi)
dev.off()


# aoa ---------------------------------------------------------------------

cl <- makePSOCKcluster(10)
registerDoParallel(cl)

gt_aoa <- aoa(model=ffs_model, 
              newdata=subset(preds, subset=ffs_model$selectedvars),
              cl=cl)

marburg_aoa <- aoa(model=ffs_model,
                   newdata=subset(marburg, subset=ffs_model$selectedvars),
                   cl=cl)

stopCluster(cl)

saveRDS(marburg_aoa, "marburg_aoa.RDS")
saveRDS(gt_aoa, "gt_aoa.RDS")

pal2 <- c("red","white")

aoa <- tm_shape(gt_aoa[[2]]) +
  tm_raster(style="cat", palette = pal2) + 
  tm_graticules(lines = FALSE) +
  tm_layout(panel.show=TRUE, panel.labels="AOA", panel.label.bg.color = "white",
            legend.outside = FALSE,
            legend.position = c("right","bottom"),
            legend.bg.color = "white",
            legend.bg.alpha = 0.8) 

di <- tm_shape(gt_aoa[[1]]) +
  tm_raster(style="quantile") + 
  tm_graticules(lines = FALSE) +
  tm_layout(panel.show=TRUE, panel.labels="DI", panel.label.bg.color = "white",
            legend.outside = FALSE,
            legend.position = c("right","bottom"),
            legend.bg.color = "white",
            legend.bg.alpha = 0.8) 


# maps --------------------------------------------------------------------

cl <- makePSOCKcluster(10)
registerDoParallel(cl)

gt_pred <- terra::predict(preds, ffs_model)
marburg_pred <- terra::predict(marburg, ffs_model)

stopCluster(cl)

saveRDS(gt_pred,"gt_pred")
saveRDS(marburg_pred, "marburg_pred")

# probability
predprob <- predict(preds, ffs_model, type="prob", index=1:length(unique(ffs_model$trainingData$.outcome))) 
names(predprob) <- levels(ffs_model$trainingData$.outcome)

predprob_df <- data.frame(as.data.frame(gt_pred),as.data.frame(predprob)) # probabilities
predprob_df$prob <- NA

for (i in unique(predprob_df$value)){
  if (is.na(i)){next()}
  predprob_df$prob[predprob_df$value==i&!is.na(predprob_df$value==i)] <- predprob_df[predprob_df$value==i&!is.na(predprob_df$value==i),i]
}

predprob$prob_all <- predprob[[1]]
values(predprob$prob_all) <- predprob_df$prob
writeRaster(predprob,"pred_prob.grd",overwrite=TRUE)

# prob for marburg
# probability
predprob_m <- predict(marburg, ffs_model, type="prob", index=1:length(unique(ffs_model$trainingData$.outcome))) 
names(predprob_m) <- levels(ffs_model$trainingData$.outcome)

predprob_m_df <- data.frame(as.data.frame(marburg),as.data.frame(predprob_m)) # probabilities
predprob_m_df$prob <- NA

for (i in unique(predprob_m_df$value)){
  if (is.na(i)){next()}
  predprob_m_df$prob[predprob_m_df$value==i&!is.na(predprob_m_df$value==i)] <- predprob_m_df[predprob_m_df$value==i&!is.na(predprob_m_df$value==i),i]
}

predprob_m$prob_all <- predprob_m[[1]]
values(predprob_m$prob_all) <- predprob_m_df$prob
writeRaster(predprob_m,"pred_prob_m.grd",overwrite=TRUE)
predprob_m <- stack("pred_prob_m.grd")
predprob <- stack("pred_prob.grd")


# maps

gt_pred1 <- gt_pred*gt_aoa$AOA 
gt_pred1[gt_pred1==0] <- NA

gt_pred1[values(predprob$prob_all)<0.5] <- NA 

gt_pred_f <- as.factor(gt_pred1)
rat <- levels(gt_pred)[[1]]
rat[["Label"]] <- gt_pred@data@attributes[[1]]$value
levels(gt_pred_f) <- rat

gt_pred_f@data@attributes[[1]] <- merge(gt_pred_f@data@attributes[[1]],
                                        luc_tab[,c("Color","Label")],
                                        by = "Label") %>% 
  dplyr::select("ID", "Label", "Color")


map_gt <- tm_shape(gt_pred_f, raster.downsample = FALSE) +
  tm_raster(palette = gt_pred_f@data@attributes[[1]]$Color,title = "LULC", 
            colorNA = "black", textNA = "high uncertainity")+
  tm_scale_bar(bg.color="white")+
  tm_grid(n.x=4,n.y=4,projection="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
          lines = FALSE)+ 
  tm_graticules(lines = FALSE) +
  tm_layout(panel.show=TRUE, panel.labels="Predicted LULC", panel.label.bg.color = "white",
            legend.position = c("left","bottom"),
                                          legend.bg.color = "white",
                                          bg.color="black",
                                          legend.bg.alpha = 0.8)+
  tm_add_legend(type = "fill",
                col="black",
                labels = "outside AOA")

tmap_save(map_gt, "LUC_guetersloh_map.png")


prob_map <- tm_shape(predprob) + 
  tm_raster(title="Probability",legend.show = TRUE) +
  tm_facets(ncol = 4) +
  tm_grid(n.x=4,n.y=4,projection="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
          lines = FALSE) +
  tm_layout(
    legend.outside = FALSE,
    legend.position = c("right","bottom"),
    legend.bg.color = "white",
    legend.bg.alpha = 0.8)

tmap_save(prob_map, "prob.png")

# same for marburg

# maps

marburg_pred1 <- marburg_pred*marburg_aoa$AOA 
marburg_pred1[marburg_pred1==0] <- NA

marburg_pred1[values(predprob_m$prob_all)<0.5] <- NA 

marburg_pred_f <- as.factor(marburg_pred1)
rat <- levels(marburg_pred)[[1]]
rat[["Label"]] <- marburg_pred@data@attributes[[1]]$value
levels(marburg_pred_f) <- rat

marburg_pred_f@data@attributes[[1]] <- merge(marburg_pred_f@data@attributes[[1]],
                                        luc_tab[,c("Color","Label")],
                                        by = "Label") %>% 
  dplyr::select("ID", "Label", "Color")

saveRDS(marburg_pred_f, "marburg_pred_f")

map_marburg <- tm_shape(marburg_pred_f, raster.downsample = FALSE) +
  tm_raster(palette = marburg_pred_f@data@attributes[[1]]$Color,title = "LUC", 
            colorNA = "black", textNA = "high uncertainity")+
  tm_scale_bar(bg.color="white")+
  tm_grid(n.x=4,n.y=4,projection="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
          lines = FALSE)+ 
  tm_graticules(lines = FALSE) +
  tm_layout(panel.show=TRUE, panel.labels="Marburg", panel.label.bg.color = "white",
            legend.show = FALSE)

tmap_save(map_marburg, "LUC_marburg_map.png")


# combine all data --------------------------------------------------------

# subset of train data

predictors <- c("B02","B03","B04","B08","B05","B06","B07","B11",
                "B12","NDVI","NDVI_5x5_sd","NDVI_3x3_sd")

trainIDs <- createDataPartition(all_dat$uniquePoly,p=0.05,list = FALSE)
trainDat <- all_dat[trainIDs,]
trainDat <- trainDat[complete.cases(trainDat[,which(names(trainDat)%in%predictors)]),]

trainDat[trainDat$Label=="Heide",]$Label <- "Gruenland"
trainDat[trainDat$Label=="Gleis",]$Label <- "Strasse"

# cv-method
indices <- CreateSpacetimeFolds(trainDat, spacevar = "uniquePoly", class="Label", k=3)

# train model
cl <- makePSOCKcluster(10)
registerDoParallel(cl)

set.seed(10)
all_model <- ffs(predictors = trainDat[, predictors], 
                 response = trainDat$Label,
                 ntree = 25,
                 tuneGrid = data.frame("mtry"=2),
                 method = "rf",
                 metric = "Kappa",
                 importance = TRUE,
                 trControl = trainControl(method="cv",index=indices$index,savePredictions="all"))

stopCluster(cl)

saveRDS(all_model, "all_model.RDS")

# get test metrics
caret::confusionMatrix(all_model)

# get details about the model
all_model$finalModel
print(all_model)
plot_ffs(ffs_model = all_model)

# get variable importance
vi <- caret::varImp(all_model)
plot(vi)


# aoa
cl <- makePSOCKcluster(10)
registerDoParallel(cl)

aoa_marburg_all <- aoa(newdata=marburg, model=all_model, cl=cl)
aoa_gt_all <- aoa(newdata=subset(preds, all_model$selectedvars),
                  model=all_model,cl=cl)

stopCluster(cl)

saveRDS(aoa_marburg_all, "aoa_marburg_all")
saveRDS(aoa_gt_all, "aoa_gt_all")

# probability

predprob_m_all <- predict(marburg, model_all, type="prob", index=1:length(unique(model_all$trainingData$.outcome))) 
names(predprob_m_all) <- levels(model_all$trainingData$.outcome)

predprob_m_all_df <- data.frame(as.data.frame(marburg),as.data.frame(predprob_m_all)) # probabilities
predprob_m_all_df$prob <- NA

for (i in unique(predprob_m_all_df$value)){
  if (is.na(i)){next()}
  predprob_m_all_df$prob[predprob_m_all_df$value==i&!is.na(predprob_m_all_df$value==i)] <- predprob_m_all_df[predprob_m_all_df$value==i&!is.na(predprob_m_all_df$value==i),i]
}

predprob_m_all$prob_all <- predprob_m_all[[1]]
values(predprob_m_all$prob_all) <- predprob_m_all_df$prob
writeRaster(predprob_m_all,"pred_prob_m.grd",overwrite=TRUE)
predprob_m_all <- stack("pred_prob_m.grd")

# probability gt

predprob_all <- predict(preds, model_all, type="prob", index=1:length(unique(model_all$trainingData$.outcome))) 
names(predprob_all) <- levels(model_all$trainingData$.outcome)

predprob_all_df <- data.frame(as.data.frame(preds),as.data.frame(predprob_all)) # probabilities
predprob_all_df$prob <- NA

for (i in unique(predprob_all_df$value)){
  if (is.na(i)){next()}
  predprob_all_df$prob[predprob_all_df$value==i&!is.na(predprob_all_df$value==i)] <- predprob_all_df[predprob_all_df$value==i&!is.na(predprob_all_df$value==i),i]
}

predprob_all$prob_all <- predprob_all[[1]]
values(predprob_all$prob_all) <- predprob_all_df$prob
writeRaster(predprob_all,"predprob_all.grd",overwrite=TRUE)

# plot map
marburg_all <- terra::predict(marburg, model=all_model)
gt_all <- terra::predict(preds, model=all_model)

marburg_all_woaoa <- marburg_all * aoa_marburg_all$AOA
marburg_all_woaoa[marburg_all_woaoa==0] <- NA

marburg_all_f <- as.factor(marburg_all_woaoa)

rat <- levels(marburg_all)[[1]]
rat[["Label"]] <- marburg_all_f@data@attributes[[1]]$value
levels(marburg_all_f) <- rat

marburg_all_f@data@attributes[[1]] <- merge(marburg_all_f@data@attributes[[1]],
                                             luc_tab[,c("Color","Label")],
                                             by = "Label") %>% 
  dplyr::select("ID", "Label", "Color")

gt_all_woaoa <- gt_all* aoa_gt_all$AOA
gt_all_woaoa[gt_all_woaoa==0] <- NA

gt_all_f <- as.factor(gt_all_woaoa)

rat <- levels(gt_all)[[1]]
rat[["Label"]] <- gt_all_f@data@attributes[[1]]$value
levels(gt_all_f) <- rat

gt_all_f@data@attributes[[1]] <- merge(gt_all_f@data@attributes[[1]],
                                       luc_tab[,c("Color","Label")],
                                       by = "Label") %>% 
  dplyr::select("ID", "Label", "Color")

saveRDS(gt_all_f, "gt_all_f")
saveRDS(marburg_all_f, "marburg_all_f")

gt_all_df <- rasterToPoints(gt_all) %>% as.data.frame()
marburg_all_df <- rasterToPoints(marburg_all) %>% as.data.frame()
names(gt_all_df) <- c("x","y","ID")
names(marburg_all_df) <- c("x","y","ID")

gt_all_df <- merge(gt_all_df, gt_all_f@data@attributes[[1]], by="ID")
marburg_all_df <- merge(marburg_all_df, marburg_all_f@data@attributes[[1]], by="ID")

gt_all_df1 <- gt_all_df %>% 
  group_by(Label) %>%
  summarise(area_ha = n()/100) %>% 
  arrange(desc(area_ha), .by_group = TRUE)

marburg_all_df1 <- marburg_all_df %>% 
  group_by(Label) %>%
  summarise(area_ha = n()/100) %>% 
  arrange(desc(area_ha), .by_group = TRUE)

saveRDS(gt_all_df1, "gt_all_df1")
saveRDS(marburg_all_df1, "marburg_all_df1")

gt_all_df1$freq <- gt_all_df1$area_ha*100
marburg_all_df1$freq <- marburg_all_df1$area_ha*100

gt_all_df2 <-  merge(gt_all_df1, luc_tab, by="Label")
marburg_all_df2 <-  merge(marburg_all_df1, luc_tab, by="Label")

gt_all_df2$region <- "Gütersloh"
marburg_all_df2$region <- "Marburg"

preds_all_b <- rbind(gt_all_df2, marburg_all_df2)

(freqpl <- ggplot(preds_all_b, aes(x=reorder(Label_en, -area_ha), y=area_ha,fill=region)) +
  geom_bar(stat="identity", position = "dodge", width = 0.8) +
  xlab("") +
  ylab("area [ha]")+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        text = element_text(size=8),
        legend.text = element_text(size=5),
        legend.title = element_text(size=6),
        legend.key.size = unit(0.8,"line"),
        legend.position = c(0.8,0.8)))
  

ggsave(plot = freqpl, filename = "freq.png", width = 5, height=3, dpi = 1200)



# maps tmap
map_marburg_all <- tm_shape(marburg_all_f, raster.downsample = FALSE) +
  tm_raster(palette = marburg_all_f@data@attributes[[1]]$Color,title = "LULC", 
            colorNA = "black", textNA = "high uncertainity")+
  tm_scale_bar(bg.color="white")+
  tm_grid(n.x=4,n.y=4,projection="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
          lines = FALSE)+ 
  tm_graticules(lines = FALSE) +
  tm_layout(panel.show=TRUE, panel.labels="Marburg", panel.label.bg.color = "white",
            legend.show = FALSE)

map_gt_all <- tm_shape(gt_all_f, raster.downsample = FALSE) +
  tm_raster(palette = gt_all_f@data@attributes[[1]]$Color,title = "LULC", 
            colorNA = "black", textNA = "high uncertainity")+
  tm_scale_bar(bg.color="white")+
  tm_grid(n.x=4,n.y=4,projection="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
          lines = FALSE)+ 
  tm_graticules(lines = FALSE) +
  tm_layout(panel.show=TRUE, panel.labels="Gütersloh", panel.label.bg.color = "white",
            legend.show = FALSE)

marburg_all_f@data@attributes[[1]] <- merge(marburg_all_f@data@attributes[[1]],
                                            luc_tab, by="Label")

marburg_all_f@data@attributes[[1]] <- select(marburg_all_f@data@attributes[[1]],
                                             c("Label_en", "Color.x","ID"))

names(marburg_all_f@data@attributes[[1]]) <- c("Label_en", "Color", "ID")

(legend_lulc <- tm_shape(marburg_all_f, raster.downsample = FALSE) +
  tm_raster(palette = marburg_all_f@data@attributes[[1]]$Color,title = "LULC", 
            colorNA = "black", textNA = "high uncertainity", 
            labels = marburg_all_f@data@attributes[[1]]$Label_en) +
  tm_layout(legend.only = TRUE))



library(grid)

png("map_all.png", height=8, width=10,units="in",res=800)

grid.newpage()
page.layout <- grid.layout(nrow = 4, ncol = 5)
pushViewport(viewport(layout = page.layout))

print(map_gt, vp=viewport(layout.pos.row = 1:2, layout.pos.col = 1:2))
print(map_marburg, vp=viewport(layout.pos.row =  1:2, layout.pos.col = 3:4))
print(map_gt_all, vp=viewport(layout.pos.row =  3:4, layout.pos.col = 1:2))
print(map_marburg_all, vp=viewport(layout.pos.row =  3:4, layout.pos.col = 3:4))
print(legend_lulc, vp=viewport(layout.pos.row = 2:3, layout.pos.col = 5))
dev.off()

tmap_save(marb_gt, "marb_gt.png", height=4, width=8)




# prediction of the model error -------------------------------------------


AOA_new <- calibrate_aoa(AOA = gt_aoa,model=ffs_model)
exp_kappa <- AOA_new$AOA[[3]]
saveRDS(AOA_new,"AOA_new")

kappa_map <- tm_shape(exp_kappa[[1]]) +
  tm_raster(style="quantile", palette =  "-YlOrRd") + 
  tm_graticules(lines = FALSE) +
  tm_layout(panel.show=TRUE, panel.labels="Predicted Kappa", panel.label.bg.color = "white",
            legend.outside = FALSE,
            legend.position = c("right","bottom"),
            legend.bg.color = "white",
            legend.bg.alpha = 0.8) 

di_aoa_k <- tmap_arrange(truec,di, aoa, kappa_map, ncol = 2, asp = 1)
tmap_save(di_aoa_k, "gt_aoa_kappa1.png", height=8, width=8)

kappa_p <- AOA_new$plot
kappa_p <- as.ggplot(kappa_p)

ggsave(plot=kappa_p, filename="kappa_plot.png", dev="png")


# Abgabe Datensatz --------------------------------------------------------

hist(gt_pred_f)

# Karte (outside AOA und <50% Propability ausmaskieren) + Prediction-dataset
## Reklassifizieren (ohne Gleis & Flughafen)
## eventuell Gl?tten (Filter: immer Originalwert (nicht neuer))
## eventuell Fl?chenanteile (wenn Ziel z.B. Prozent abgestorbene Fichtenw?lder)
# Trainingsdaten, Modell & Pr?diktoren
# Sampling design + Validierungsma?e
## Kappa+Accuracy; Pro Klasse (Confusion matrix)














