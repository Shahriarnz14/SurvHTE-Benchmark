#################################################################################################
########################################## RIST      ############################################
#################################################################################################
library(survival)

Muti_ERT_fit <-function(dataset, M=50, K=5, L=3, nmin=6, SupLogRank=1, tao=25, impute=c("random", "expected"))
{
	newset=dataset
	newp=dim(dataset)[2]
	P=newp-2
	need=ceiling(dim(dataset)[1]/nmin)
	time_intrest=sort(dataset[dataset[,P+1]==1, P+2])

	Forest=matrix(0,2*need,7*M)
	Large=rep(0,M)
	Forestdata=matrix(0,dim(dataset)[1],M)
	SurvMat=list()

	for (i in 1:M) #fit M trees
	{
		Treefit=Fit_ERT_Tree(newset,dataset, K, nmin, SupLogRank, Step=0, tao)
		Forest[1:dim(Treefit$Tree)[1],(7*(i-1)+1):(7*i)]=Treefit$Tree
		Large[i]=dim(Treefit$Tree)[1]
		Forestdata[,i]=Treefit$UpdateData[,dim(dataset)[2]]
		SurvMat[[i]]=Treefit$Survival_All
	}

	if (sum(newset[, P+1] == 0) > 0)
	{
		censornum=seq(1,dim(dataset)[1])[dataset[,P+1]==0]

		censorset = dataset[dataset[,P+1]==0,]

		censor_surv = Muti_ERT_Predict(censorset, Forest, SurvMat, time_intrest)$Surv_predict

		cond_sample_set=list()
		cond_sample_rate=list()

		len=length(time_intrest)

		for (q in 1:dim(censorset)[1])
		{
			sub_surv=c(1,censor_surv[q,])
			locate=sum(time_intrest<=censorset[q,P+2])+1

			con_surv1=sub_surv[locate:length(sub_surv)]
			con_surv2=c(1,con_surv1)-c(con_surv1,0) 
			con_surv=con_surv2[2:length(con_surv2)]
			cond_sample_rate[[q]]=con_surv/sum(con_surv)

			cond_sample_set[[q]]=c(time_intrest[time_intrest>censorset[q,P+2]],tao)
		}
	}
	
	Forest_0=Forest
	SurvMat_0=SurvMat

	if ((sum(newset[, P+1] == 0) == 0) & (L > 0))
	{
		warning("There is no censored observation to impute, set L to 0")
		L = 0
	}
	
	Forest_seq=list()
	SurvMat_seq=list()

	if (L>0)
	{
		for (j in 1:L)
		{
			Forest=matrix(0,3*need,7*M)
			Large=rep(0,M)
			Forestdata=matrix(0,dim(dataset)[1],M)
			SurvMat=list()

			for (i in 1:M) #fit M trees after imputation
				{
					newobtime=rep(0,dim(censorset)[1])
					newobtime2=rep(0,dim(censorset)[1])

					for (q in 1:dim(censorset)[1])
					{
						samplelong=length(cond_sample_rate[[q]])
						if (cond_sample_rate[[q]][samplelong]==1)
						{
							newobtime[q]=tao
							newobtime2[q]=tao
						}else
						{newsample=rmultinom(1,size=1, prob= cond_sample_rate[[q]])
						newobtime[q]=cond_sample_set[[q]][newsample==1]

						samplerate=cond_sample_rate[[q]][1:(samplelong-1)]/sum(cond_sample_rate[[q]][1:(samplelong-1)])
						exptime=sum(cond_sample_set[[q]][1:(samplelong-1)]*samplerate)
						newsample2=rbinom(1, 1, prob=cond_sample_rate[[q]][samplelong])
						newobtime2[q]=tao*(newsample2==1)+exptime*(newsample2==0)
						}
					}
					
					if (impute=="random")
					{
						newcensorind=1-(newobtime==tao)
						newset[censornum,P+1]=newcensorind
						newset[censornum,P+2]=newobtime
					}

					if (impute=="expected")
					{
						newcensorind=1-(newobtime2==tao)
						newset[censornum,P+1]=newcensorind
						newset[censornum,P+2]=newobtime2
					}

					Treefit=Fit_ERT_Tree(newset,dataset, K, nmin, SupLogRank, Step=1 ,tao)
					Forest[1:dim(Treefit$Tree)[1],(7*(i-1)+1):(7*i)]=Treefit$Tree
					Large[i]=dim(Treefit$Tree)[1]
					Forestdata[,i]=Treefit$UpdateData[,dim(dataset)[2]]
					SurvMat[[i]]=Treefit$Survival_All
				}

			censor_surv = Muti_ERT_Predict(censorset, Forest, SurvMat, time_intrest)$Surv_predict

			cond_sample_set=list()
			cond_sample_rate=list()

			len=length(time_intrest)

			for (q in 1:dim(censorset)[1])
			{
				sub_surv=c(1,censor_surv[q,])
				locate=sum(time_intrest<=censorset[q,P+2])+1

				con_surv1=sub_surv[locate:length(sub_surv)]
				con_surv2=c(1,con_surv1)-c(con_surv1,0) 
				con_surv=con_surv2[2:length(con_surv2)]
				cond_sample_rate[[q]]=con_surv/sum(con_surv)

				cond_sample_set[[q]]=c(time_intrest[time_intrest>censorset[q,P+2]],tao)
			}

			Forest_seq[[j]]=Forest
			SurvMat_seq[[j]]=SurvMat
		}
	}

return(list("time_intrest"=time_intrest,
		"Forest_seq"=Forest_seq,
		"SurvMat_seq"=SurvMat_seq,
		"Forest_0"=Forest_0,
		"SurvMat_0"=SurvMat_0))

}


Muti_ERT_Predict <- function (testset, Forest, SurvMat, time_intrest)
{

XP=dim(testset)[2]-2
M=dim(Forest)[2]/7
newsubject=testset[,1:XP]
newn=dim(newsubject)[1]
TMT_predict=rep(0,newn)
Surv_predict=matrix(0,newn,length(time_intrest))
TMT_temp=rep(0,M)
Surv_temp=matrix(0, M, length(time_intrest))

for (s in 1:newn)
{
	new_cov=newsubject[s,]
	
	for (i in 1:M)
		{
			Tree=Forest[,(7*(i-1)+1):(7*i)]
			Whole_predict=Evaluate_X(new_cov, Tree, SurvMat[[i]])
			TMT_temp[i]=Whole_predict$TMT_pre
			Surv_temp[i,]=Whole_predict$Surv_pre
		}
	TMT_predict[s]=mean(TMT_temp)
	Surv_predict[s,]=colMeans(Surv_temp)
}

return(list("TMT_predict"=TMT_predict,
		"Surv_predict"=Surv_predict,
		"time_intrest"=time_intrest))
}




################### build a tree ##############

Fit_ERT_Tree <- function (dataset, dataset2, K, nmin, SupLogRank=0, Step, tao)  #nmin is the minimal number of observed data 
{
	data_n <<- dim(dataset)[1]

	One_Tree <<- matrix(0,2*data_n/nmin,7)        # Global variable to save the tree
	BookingTable <<- matrix(0, 2*data_n/nmin ,1)  # Global variable to save used locations
	BookingTable[1,1] <<- 1
	P <<- dim(dataset)[2]-2
	Tao <<- tao
	obs <- seq(1,dim(dataset)[1])
	rate <- rep(1, dim(dataset)[1])

	time_points <<- sort(dataset2[dataset2[, (P+1)]==1, (P+2) ])
	Survival_All <<- matrix(0, 2*data_n/nmin , length(time_points))

	dataset <- cbind(dataset,obs)
	backset <<- cbind(dataset2,obs,rate)
	
	Further_Split_A_Node(dataset, K, nmin, 1, SupLogRank, Step)
	
	output <- list()
	output$Tree <- One_Tree[BookingTable[,1]==1,]
	output$UpdateData <- backset
	output$Survival_All <- Survival_All[BookingTable[,1]==1,]
	out <- output
}

################## split a node ################
#   use functions: Stop_Split, Find_A_Split,   #
################################################

Further_Split_A_Node <- function (subset, K, nmin, node, SupLogRank, Step)
{

	Split_result=Stop_Split(subset, nmin)   
	if (Split_result$Stop==1)
	{
    	One_Tree[node, 1]<<-NA
    	One_Tree[node, 2]<<-NA
    	One_Tree[node, 3]<<-NA
    	One_Tree[node, 4]<<-NA
	
	UpdateCensor(subset, node, Step)

	One_Tree[node, 7]<<-mean(subset[,P+2])      ########## mean of expected observed time (consideing survival function) 
	}
	
	else
	{
		split <- Find_A_Split(subset, Split_result$SplitVariable, Split_result$SplitValue, SupLogRank)  
		
    		a <- split[1]
    		ac <- split[2]

		NAobs=(is.na(subset[,a]>=ac)==1)

		noneNALeft=(subset[,a]<ac)
		noneNARight=(subset[,a]>=ac)
		noneNALeft[is.na(noneNALeft)==1]=0
		noneNARight[is.na(noneNARight)==1]=0

		Left_size=sum(subset[,a]<ac, na.rm = TRUE)
		Right_size=sum(subset[,a]>=ac, na.rm = TRUE)
		goleft=rbinom(length(NAobs),1,NAobs*Left_size/(Left_size+Right_size))
		goright=NAobs-goleft

		Left_obs=noneNALeft+goleft
		Right_obs=noneNARight+goright

		subsetL=subset[Left_obs==1,]
		subsetR=subset[Right_obs==1,]

    ###########  we search for a place for left tree:

    		LeftNode = node+1;
    		while (BookingTable[LeftNode,1]==1)
    			LeftNode=LeftNode+1

    		BookingTable[LeftNode,1] <<- 1
    
    ############ we search a place for right tree:

    		RightNode = LeftNode+1;
    		while (BookingTable[RightNode,1]==1)
			RightNode=RightNode+1;
	
    		BookingTable[RightNode,1] <<- 1;
    

	########### finishing building the current node

    		One_Tree[node, 1] <<- a;
    		One_Tree[node, 2] <<- ac;
		One_Tree[node, 3] <<- LeftNode;
		One_Tree[node, 4] <<- RightNode;

		One_Tree[node, 5] <<- dim(subsetL)[1];
		One_Tree[node, 6] <<- dim(subsetR)[1];
		One_Tree[node, 7] <<- NA;	
	    	#One_Tree[node, 7] <<- dim(subset)[1];	

		Further_Split_A_Node(subsetL, K, nmin, LeftNode, SupLogRank, Step);
    
		Further_Split_A_Node(subsetR, K, nmin, RightNode, SupLogRank, Step);	
	}
}



UpdateCensor <- function (subset, node, Step)
{
	if (Step==0)
		{
			#calculate KM estimate of failset 
			#need output for further use
			Survival_All[node,] <<- KM_func(subset)
		}
	if (Step==1)
		{
			#calculate emperical estimate of failset
			#need output for further use
			obpoints=subset[,P+2]
			for (i in 1:length(time_points))
				{
				Survival_All[node,i] <<- mean(obpoints>time_points[i]) 
				}
		}

}


KM_func <- function(subset)
{
	failset=subset[,(P+1):(P+2)]
	uniq=sort(unique(failset[failset[,1]==1,2]))
	nuniq=length(uniq)

	d=rep(0,nuniq)
	y=rep(0,nuniq)

	failset2=failset[failset[,1]==1,2]
	
	for (i in 1:nuniq)
		{
		d[i]=sum(failset2==uniq[i])
		y[i]=length(failset[failset[,2]>=uniq[i],2])
		}	
	Survive=c(1,cumprod(1-d/y))
	Surv_all=rep(0, length(time_points))
	for (j in 1:length(time_points))
		{
		Surv_all[j] <- Survive[sum(time_points[j]>=uniq)+1] 
		}
	return(Surv_all)
}


################# stop a spliting #################

Stop_Split <- function(subset, nmin)
{
	nobs=sum(subset[,P+1])     # number of observed data
	if (nobs<(nmin*2)) {out<-list("Stop"=1,"SplitVariable"=NA, "SplitValue"=NA)} else    
	{
		obsset=subset[subset[,P+1]==1,]
		sampleMin=rep(0,P)
		sampleMax=rep(0,P)


		for (i in 1:P)
		{
			sampleMin[i]=sort(obsset[,i])[nmin]
			sampleMax[i]=tail(sort(obsset[,i]),nmin)[1]
		}
		
		stop=1-(sum(sampleMin<sampleMax)>0)

		if (stop==1){out<-list("Stop"=1, "SplitVariable"=NA, "SplitValue"=NA)} else
		{
			cansample=seq(1,P)[sampleMin<sampleMax]
			split=rep(NA,P)
			for (i in cansample)
			{
				split[i]=runif(1, sampleMin[i], sampleMax[i])
			}

			ransample=cansample
			if (K<=length(cansample))
			{
				ransample=sample(cansample,K)
			}
			out<-list("Stop"=0,"SplitVariable"=ransample, "SplitValue"=split[ransample])
		}
	}
}

################# Find a split ###############

Find_A_Split <- function(subset, ransample, splituse, SupLogRank)
{
	lp=length(ransample)
	score0=rep(1, lp)
	score1=rep(1, lp)
	score2=rep(1, lp)

	for (j in 1:lp)
		{
			if (SupLogRank==2)
				{
					Group=(subset[,ransample[j]]<splituse[j])+1	
					Censor=subset[,(P+1)]
					Y=subset[,(P+2)]
					NAobs=is.na(Group)
					Group=Group[NAobs==0]
					Censor=Censor[NAobs==0]
					Y=Y[NAobs==0]
		 			score2[j]=surv.Rtest(Y, Censor, Group, 0, 0, FALSE, error=1.0e-8)$p   #p-value small is better
				} 
			if (SupLogRank==1)
				{
					Group=(subset[,ransample[j]]<splituse[j])+1	
					Censor=subset[,(P+1)]
					Y=subset[,(P+2)]
					if (min(Y)==max(Y)) 
						{score1[j]=1} else
							{
							LRtest=survdiff(Surv(Y,Censor)~Group)
							score1[j]=1-pchisq(LRtest$chisq,1)
							}
				} 

			if (SupLogRank==0)
				{
					score0[j]=treescore(subset,ransample[j],splituse[j])
				} 

		}

	if (SupLogRank==0)
	{
		os=order(score0, decreasing = TRUE)
		Sval=splituse[os]
		Svar=ransample[os]
		out_s<-c(Svar[1],Sval[1])          	
	}

	if (SupLogRank==1)
	{
		os=order(score1)
		Sval=splituse[os]
		Svar=ransample[os]
		out_s<-c(Svar[1],Sval[1])         			
	} 

	if (SupLogRank==2)
	{
		os=order(score2)
		Sval=splituse[os]
		Svar=ransample[os]
		out_s<-c(Svar[1],Sval[1])         			
	} 

	return(out_s)
}


treescore <- function(subset, selectvar, split)
{
	n=dim(subset)[1]
	totalvar=var(subset[,P+2])
	leftY=subset[(subset[,selectvar]<split),(P+2)]
	rightY=subset[(subset[,selectvar]>split),(P+2)]
	leftvar=max(c(0, var(leftY, na.rm = TRUE)), na.rm = TRUE)
	rightvar=max(c(0, var(rightY, na.rm = TRUE)), na.rm = TRUE)
	sl=length(na.omit(leftY))
	sr=length(na.omit(rightY))
	out<- (totalvar - sl*leftvar/n - sr*rightvar/n)/totalvar
}


################################ evaluating ###########################

Evaluate_DataX <- function(X, Tree, SurvMat)
{
	n=dim(X)[1]
	y=rep(0,n)
	for (i in 1:n)
	{
		y[i]=Evaluate_X(X[i,],Tree, SurvMat)
	}
	out<-y
}

Evaluate_X <- function(X, Tree, SurvMat)
{
	if(is.matrix(Tree)==0)
		{out<- list("TMT_pre"=Tree[7], "Surv_pre"=SurvMat)} else
			out<- Evaluate_It(X, Tree, 1, SurvMat)
}



Evaluate_It <- function(X, Tree, node, SurvMat)
{
	if (is.na(Tree[node,2])==1)
	{out <- list("TMT_pre"=Tree[node,7],"Surv_pre"=SurvMat[node,]) } else
		{
			if (is.na(X[Tree[node,1]]))
				{
				GoRight = rbinom( 1, 1, Tree[node,6]/( Tree[node,6] + Tree[node, 5]) )
				} else 
					{
					GoRight=(X[Tree[node,1]]>=Tree[node,2])
					}
		out<- Evaluate_It(X, Tree, Tree[node,3]+GoRight, SurvMat)
		}
}


#############################################################################################################
#############################################################################################################

#                                       Sup Log-Rank Test

#############################################################################################################
#############################################################################################################



sup.G<-function(x,m=10)
{
     k<-m:0
     (4/pi)*sum(((-1)^k)/(2*k+1)*exp(-(pi^2)*((2*k+1)^2)/(8*x^2)))
}
sup.g<-function(x,m=10)
{
     k<-m:0
     (pi/x^3)*sum(((-1)^k)*(2*k+1)*exp(-(pi^2)*((2*k+1)^2)/(8*x^2)))
}
cnorm<-function(z,thresh=3.6,delta=0.6,kk=4){
check<-F
if(z<0){
     z<-(-1)*z
     check<-T
}
if(z<thresh){
     out<-1-pnorm(z)
}
else{
     term<-1
     tally<-term
     if(kk>1){
          for(k in 1:(kk-1)){
               term<-(-1)*term*(2*k-1)/z^2
               tally<-tally+term
          }
     }
     out<-tally*dnorm(z)/z
     if(z<thresh+delta){
          x<-1-pnorm(z)
          out<-x+(z-thresh)*(out-x)/delta
     }
}
if(check){out<-1-out}
out
}
sup.inverse<-function(alpha,error=1e-8)
{
     x<-qnorm(1-alpha/4)
     temp<-max(1,2/x)
     m<-ceiling((x/pi)*sqrt(2*log(temp/(pi*error)))-0.5)
     if(m<0){m<-0}
     interror<-1
     while(interror>error)
     {
          yx<-sup.G(x,m=m)
          dg<-sup.g(x,m=m)
          delta<-(1-alpha-yx)/dg
          x<-x+delta
          interror<-sup.G(x)-(1-alpha)
     }
     x
}
sup.r<-function(alpha, beta, error=1e-8)
{
     u<-sup.inverse(alpha,error=error)
     y<-1-beta
     ml<-qnorm(1-alpha/2)+qnorm(1-beta)
     x<-ml
     delta<-1
     while(delta>error)
     {
          yx<-cnorm(u-x)+exp(2*x*u)*cnorm(u+x)
          dp<-dnorm(u-x)-exp(2*u*x)*dnorm(u+x)+2*u*exp(2*u*x)*cnorm(u+x)
          delta<-(y-yx)/dp
          x<-x+delta
     }
     (x/ml)^2    
}
surv.Rtest<-function (time, delta, group, rho=0, gamma=0, logrank=F, 
     error=1.0e-8) 
{    
     otime <- order(time)
     time <- time[otime]
     delta <- delta[otime]
     n<-length(time)
     if((rho+gamma)==0){
          weight<-rep(1,n)
     }
     else{
          km.left<-KM.left(time,delta)
          weight<-km.left^rho*(1-km.left)^gamma
     }
     group <- group[otime] - 1
     n2 <- sum(group)
     atrisk2 <- n2 - cumsum(group) + group
     n1 <- n-n2
     atrisk1 <- n1 - cumsum(1 - group) + 1 - group
     delta1 <- delta * (1 - group)
     delta2 <- delta * group
     y1 <- tapply(atrisk1, time, "max")
     y2 <- tapply(atrisk2, time, "max")
     d1 <- tapply(delta1, time, "sum")
     d2 <- tapply(delta2, time, "sum")
     weight<-tapply(weight,time,"max")
     w <- (y1 * y2)/(y1 + y2)
     terms <- (d1/y1 - d2/y2)[w > 0]
     temp<-y1+y2-1
     temp<-ifelse(temp<1,1,temp)
     cc<-1-(d1+d2-1)/temp
     vterms <- (cc*(d1 + d2)/(y1 + y2))[w > 0]
     weight<-weight[w > 0]
     w <- w[w > 0]
     terms <- (weight * w * terms)/sqrt(sum(weight^2 * w * vterms))
     temp<-c(0,cumsum(terms))
     xs<-max(temp, na.rm = TRUE)   # changed from xs<-max(temp) to deal with missing values
     xi<-min(temp, na.rm = TRUE)   # changed from xi<-min(temp) to deal with missing values
     if(abs(xs)>abs(xi)){test<-xs}
     else{test<-xi}
     x <- abs(test)
     m<-ceiling(max(c(1,(x*sqrt(2)/pi)*sqrt(max(c(1,log(1/(pi*error)))))-0.5)))
     p<-1-sup.G(x,m=m)
     out <- NULL
     out$test <- test
     out$p <- p
     if(logrank){
          x<-temp[length(temp)]
          out$test.logrank<-x
          out$p.logrank<-2*cnorm(abs(x))
     }
     out
}
KM.left<-function(time,delta){
     n<-length(time)
     dtime<-tapply(time,time,"max")
     ddelta<-tapply(delta,time,"sum")
     dy<-tapply(rep(1,n),time,"sum")
     m<-length(dy)
     y<-rep(n,m)-c(0,cumsum(dy)[1:(m-1)])
     km<-1
     km.left<-rep(0,m)
     for(i in 1:m){
          km.left[i]<-km
          km<-km*(1-ddelta[i]/y[i])
     }
     out<-rep(0,n)
     for(i in 1:n){
          out[i]<-min(km.left[dtime==time[i]])
     }
     out
}




