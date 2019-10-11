library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('grid') # visualisation
library('gridExtra') # visualisation
library('corrplot') # visualisation
library('ggrepel') # visualisation
library('RColorBrewer') # visualisation
library('data.table') # data manipulation
library('dplyr') # data manipulation
library('readr') # data input
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('lazyeval') # data wrangling
library('broom') # data wrangling
library("tibble")# data wrangling
library('stringr') # string manipulation
library('purrr') # string manipulation
library('forcats') # factor manipulation
library('lubridate') # date and time
library('forecast') # time series analysis
library('prophet') # time series analysis
library('lazyeval')

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

train <- fread("C:/Users/Galytix/Downloads/web-traffic-time-series-forecasting/train_1.csv",
              stringsAsFactors = F, check.names = F)
train <- as.data.frame(train)
key <- fread("C:/Users/Galytix/Downloads/web-traffic-time-series-forecasting/key_1.csv",
             stringsAsFactors = F, check.names = F)
key <- as.data.frame(key)

# Missing Values 
sum(is.na(train))/(ncol(train)*nrow(train))
# splitting the data 
tdates <- train[,!colnames(train) %in% "Page"]
foo <- train %>% select(Page) %>% rownames_to_column()
mediawiki <- foo[grepl("mediawiki",foo$Page,ignore.case = F),]
wikimedia <- foo[grepl("wikimedia",foo$Page, ignore.case = F),]
wikipedialogical <- grepl("wikipedia",foo$Page, ignore.case = F) &
  !grepl("wikimedia",foo$Page, ignore.case = F) & !grepl("mediawiki",foo$Page, ignore.case = F)
wikipedia <- foo[wikipedialogical,]
wikipedia <- wikipedia %>% separate(Page, into = c("foo","bar"),sep = ".wikipedia.org_") %>% 
      separate(foo,into = c("article","locale"), sep = -3) %>% 
      separate(bar, into = c("access","agent"),sep = "_") %>% mutate(locale=str_sub(locale,2,3))
wikimedia <- wikimedia %>% separate(Page, into = c("article","bar"), sep="_commons.wikimedia.org_") %>% 
  separate(bar, into = c("access","agent"),sep = "_") %>% add_column(locale="wikmed")

mediawiki <- mediawiki %>% separate(Page, into = c("article", "bar"), sep = "_www.mediawiki.org_") %>%
  separate(bar, into = c("access", "agent"), sep = "_") %>% add_column(locale = "medwik")
tpages <-  wikipedia %>% full_join(wikimedia, by=c("rowname","article","locale","access","agent")) %>% 
  full_join(mediawiki, by=c("rowname","article","locale","access","agent"))

extract_ts <- function(rownr){
  dataTime <- tdates[rownr,]
  outDf <- data.frame(dates=colnames(dataTime),views=as.numeric(df[1,]))
  return(outDf)
}
extract_ts_nrm <- function(rownr){
  dataTime <- tdates[rownr,]
  outDf <- data.frame(dates=colnames(dataTime),views=as.numeric(df[1,]))
  outDf <- outDf %>% mutate(views=views/mean(views))
  return(outDf)
}
plot_rownr <- function(rownr){
  art <- tpages[rownr,]$article
  loc <- tpages[rownr,]$locale
  acc <- tpages[rownr,]$access
  extract_ts(rownr) %>% ggplot(aes(x=dates,y=views))+geom_line()+
    geom_smooth(method = "loess", color="blue", span=1/5)+
    labs(title = paste(art,"-",loc,"-",acc))
}
plot_rownr_log <- function(rownr){
  art <- tpages[rownr,]$article
  loc <- tpages[rownr,]$locale
  acc <- tpages[rownr,]$access
  extract_ts_nrm(rownr) %>% ggplot(aes(x=dates, y=views))+ geom_line()+
    geom_smooth(method = "loess", color="red", span=1/5)+
    labs(title = paste(art,"-",loc,"-",acc))+
    scale_y_log10()+labs(y="log views")
  
}
plot_rownr_zoom <- function(rownr,start,end){
  art <- tpages[rownr,]$article
  loc <- tpages[rownr,]$locale
  acc <- tpages[rownr,]$access
  data1 <- extract_ts(rownr)
  data1 <- data1[data1$dates > ymd(start) & data1$dates <= ymd(end),]
  data1 %>% ggplot(aes(x=dates, y=views))+ geom_line()+
    geom_smooth(method = "loess", color="red", span=1/5)+
    labs(title = paste(art,"-",loc,"-",acc))
  
}
# plot_rownr(11214)
plot_names <-  function(art, acc, agent){
 vec <- tpages[tpages$article %in% art & tpages$access %in% acc & tpages$agent %in% agent,"rowname"]
 selecttpages <- tpages[tpages$rowname %in% vec,]
 td <- tdates[vec,] %>% rownames_to_column("rowname")
 df <- merge(td,selecttpages,by="rowname")
 df <- reshape2::melt(df,id.vars=c("rowname","article","locale","access","agent"))
 df$locale <- as.factor(df$locale)
 df$variable <- as.Date(as.character(df$variable),"%Y-%m-%d")
 df %>% ggplot(aes(x=variable,y=value,color=locale))+geom_line()
}
plot_names_nrm <-  function(art, acc, agent){
  vec <- tpages[tpages$article %in% art & tpages$access %in% acc & tpages$agent %in% agent,"rowname"]
  selecttpages <- tpages[tpages$rowname %in% vec,]
  td <- tdates[vec,] %>% rownames_to_column("rowname")
  df <- merge(td,selecttpages,by="rowname")
  df <- reshape2::melt(df,id.vars=c("rowname","article","locale","access","agent"))
  df$locale <- as.factor(df$locale)
  df$variable <- as.Date(as.character(df$variable),"%Y-%m-%d")
  df %>% ggplot(aes(x=variable,y=value,color=locale))+geom_line()+scale_y_log10()
}




# p1 <- tpages %>% ggplot(aes(agent))+geom_bar(fill="red")
# p2 <- tpages %>% ggplot(aes(access))+geom_bar(fill="red")
# p3 <- tpages %>% ggplot(aes(locale, fill=locale))+geom_bar()+theme(legend.position = "none")
# layout <- matrix(c(1,2,3,3),2,2, byrow = T)
# multiplot(p1,p2,p3,layout = layout)
# extract_ts(rownr = 10)






















