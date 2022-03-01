CoxLikBarycentricPointSpread.prepare.new3.nsub <- function(OmegaMesh,A_SpatialPolygons,nsub=NULL){
  
  dt0 <- data.table(x=OmegaMesh$loc[,1],y=OmegaMesh$loc[,2])
  
  # Creating new integration points and associated weights with the 28th june 2019 method implemented in inlabru
  ips <- inlabru::ipoints(sampler = A_SpatialPolygons, domain = OmegaMesh, int.args = list(nsub1 = nsub))
  
  # Matching the 
  dt <- data.table::data.table(weight=ips$weight,sp::coordinates(ips))
  dt[,weight := sum(weight),by=.(x,y)]
  dt <- unique(dt)
  
  ret <- merge(dt0,dt,by=c("x","y"),all.x=T,all.y=T,sort=F) # Important not to sort the outpout DT here
  ret[is.na(weight),weight:=0]
  
  if(!all.equal(ret[,.(x,y)],dt0)){
    stop("weight vector does not match mesh locations")
  } else {
    return(ret$weight)
  }
}