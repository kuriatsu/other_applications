#!/usr/bin/env Rscript

# define data
Obs <- c(1,2,3,4,5,6)
X1 <- c(1,1,0,5,6,4)
X2 <- c(4,3,4,1,2,0)
sample_df = data.frame(Obs=Obs, X1=X1, X2=X2)
sample_df

# (a) plot the observation
png(paste(c("result", 0, ".png"), collapse=""))
plot(sample_df$X1, sample_df$X2, xlim=c(0,6), ylim=c(0,6), col="black", pch=16, cex=1.5)

# (b) randomly assign a cluster label to each observation with sample()
cluster_list <- sample(2,6,replace=TRUE)
sample_df <- cbind(sample_df, cluster=cluster_list)

# report the label for each observation
sample_df

# (e) repert (c) and (d) 10 times
for (iteration in 1:10)
{
	# (c) compute the centroid for each cluster
	centroid_X1 <- mean(sample_df[sample_df[ , 4] == 1,]$X1)
	centroid_X2 <- mean(sample_df[sample_df[ , 4] == 1,]$X2)
	centroid_1 <- c(centroid_X1, centroid_X2)

	centroid_X1 <- mean(sample_df[sample_df[ , 4] == 2,]$X1)
	centroid_X2 <- mean(sample_df[sample_df[ , 4] == 2,]$X2)
	centroid_2 <- c(centroid_X1, centroid_X2)

	# (f) visualize & color the observations according to the cluster labels obtained.
	png(paste(c("result", iteration, ".png"), collapse=""))
	plot(sample_df[sample_df$cluster == 1,]$X1, sample_df[sample_df$cluster == 1,]$X2, xlim=c(0,6), ylim=c(0,6), col="#00AFBB", ann=F, pch=16, cex=1.5)
	par(new=T)
	plot(sample_df[sample_df$cluster == 2,]$X1, sample_df[sample_df$cluster == 2,]$X2, xlim=c(0,6), ylim=c(0,6), col="#FC4E07", pch=16, cex=1.5)
	par(new=T)
	plot(c(centroid_1[1], centroid_2[1]), c(centroid_1[2], centroid_2[2]), col="#E7B800", xlim=c(0,6), ylim=c(0,6), pch=4, cex=1.5)

	# (d) assign each observation to the centroid to which it is closest, in terms of Euclidean distance.
	cluster_list <- list()
	for (obs_num in 1:6)
	{
		dist_cluster_1 <- sqrt((sample_df[obs_num, 2] - centroid_1[1]) ^ 2 + (sample_df[obs_num, 3] - centroid_1[2]) ^ 2)
		dist_cluster_2 <- sqrt((sample_df[obs_num, 2] - centroid_2[1]) ^ 2 + (sample_df[obs_num, 3] - centroid_2[2]) ^ 2)

		if (dist_cluster_1 <= dist_cluster_2)
		{
			sample_df[obs_num, 4] <- c(cluster_list, 1)
		}
		else
		{
			sample_df[obs_num, 4] <- c(cluster_list, 2)
		}
	}

	# report the cluster labels for each observation
	print(sample_df)
}
