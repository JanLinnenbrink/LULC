rm(list = ls())

library(raster)
library(stars)
library(sf)
library(RStoolbox)
library(terra)
library(dplyr)
library(mapview)
library(caret)
library(tmap)
library(terrainr)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))


# load data ---------------------------------------------------------------

rs <- st_read("train_data.gpkg") %>% st_transform(st_crs(ug))
ug <- st_read("ug.shp") 

s2_10 <- lapply(c(paste0("B",c(2:4,8),".jp2")), raster) %>% stack %>% crop(ug)
s2_20 <- lapply(c(paste0("B",c(5:7,"8A",11:12),".jp2")), raster) %>% stack %>% crop(ug)
s2_40 <- lapply(c(paste0("B",9:10,".jp2")), raster) %>% stack %>% crop(ug)

s2_202 <- resample(s2_20, s2_10)
s2 <- stack(s2_10$B2, s2_10$B3, s2_10$B4, s2_202$B5, s2_202$B6, s2_202$B7,
               s2_10$B8, s2_202$B8A, s2_202$B11, s2_202$B12)

rm(list=c("s2_10","s2_20", "s2_202", "s2_40"))

# rescale bands to 0-255 (visualization)
rescale_new <- function(x, x.min = NULL, x.max = NULL, new.min = 0, new.max = 1) {
  if(is.null(x.min)) x.min = min(x)
  if(is.null(x.max)) x.max = max(x)
  new.min + (x - x.min) * ((new.max - new.min) / (x.max - x.min))
}

s2l <- as.list(s2)

s2r <- lapply(s2l, function(x) {rescale_new(x, x.min = cellStats(x, "min"), 
                                                     x.max = cellStats(x, "max"),
                                            new.min = 0, new.max=255)})

s2s <- brick(s2r)
s2_stars <- st_as_stars(s2s)

s2rgb <- stars::st_rgb(s2_stars[,,,3:1],
                       dimension = 3,
                       maxColorValue = 255,
                       use_alpha = FALSE, 
                       probs = c(0.02, 0.98), 
                       stretch = TRUE)

s2nir <- stars::st_rgb(s2_stars[,,,c(7,3,2)],
                       dimension = 3,
                       maxColorValue = 255,
                       use_alpha = FALSE, 
                       probs = c(0.02, 0.98), 
                       stretch = TRUE)

truec <- tm_shape(s2rgb) + 
  tm_raster() +
  tm_graticules(lines = FALSE) +
  tm_layout(panel.show=TRUE, panel.labels="True color composite", panel.label.bg.color = "white") #+
  #tm_shape(rs) +
  #tm_polygons(col = "yellow")


falsec <- tm_shape(s2nir) + tm_raster() + tm_graticules(lines = FALSE) +
  tm_layout(panel.show=TRUE, panel.labels="False color composite", panel.label.bg.color = "white") +
  tm_shape(rs) +
  tm_polygons(col = "yellow")

tr_fc_a <- tmap_arrange(truec, falsec, ncol = 2, asp = 1)

tmap_save(tr_fc_a, "True_false_color_c.png", height=4, width=8)



# prepare predictors ------------------------------------------------------

ndvi <- overlay(s2$B4, s2$B8, fun=function(r,nir) {n <- (nir-r)/(nir+r); return(n)})

mlist <- list(matrix(1,3,3),matrix(1,5,5))

ndvi_sd <- lapply(mlist, focal, x=ndvi, fun=sd)


preds <- stack(s2[[1]], s2[[2]], s2[[3]], s2[[7]],
               s2[[4]], s2[[5]], s2[[6]], s2[[8]],
               s2[[10]], ndvi, ndvi_sd[[1]], ndvi_sd[[2]])

names(preds) <- c("B02","B03","B04","B08","B05","B06","B07","B11",
                  "B12","NDVI","NDVI_3x3_sd","NDVI_5x5_sd")


writeRaster(preds, "preds.grd", overwrite = TRUE)

preds_l <- as.list(preds)

preds_rs <- lapply(preds_l, function(x) {rescale_new(x, x.min = cellStats(x, "min"), 
            x.max = cellStats(x, "max"),new.min = 0, new.max=255)})

preds_rs_s <- stack(preds_rs)

saveRDS(preds_rs_s, "preds_resc")
saveRDS(preds, "predictors")


# combine with the response

rs$ID <- c(1:length(rs$geometry))
rs$id <- NULL
rs[rs$Label=="Gleis",]$Label <- "Strasse"
rs[rs$Label=="Heide",]$Label <- "Gruenland"

pr_rs <- full_join(terra::extract(x=preds, y=rs, df=TRUE), st_drop_geometry(rs), by="ID")
pr_rs$Region <- "Guetersloh"

saveRDS(pr_rs,"Trainingsdaten_Guetersloh_Jan.RDS")

