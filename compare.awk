BEGIN{FS=","}
{
        if(NR == 3) 
        {
                for(var = 1; var <= NF; var++)
                        firstrow[var] = $var;
        }
        if(NR == 4)
        {
                for(var = 1; var <= NF; var++)
                        secondrow[var] = $var;
        }
}
END{
        for(var in firstrow)
           actual[firstrow[var]]++;  

        print "Difference between predicted and actual label";  
        for(i = 0;i < 10; i++)
        {
                print "For digit", i;
                print  "\tActual", actual[i];   
                
                for(j = 0; j < 10; j++)
                   predicted[j]=0;                      

                for(var in secondrow)
                {
                    if(firstrow[var] == i)
                      predicted[secondrow[var]]++;      
                } 

                print "\tPredicted";
                for(j = 0; j < 10; j++)
                  print "\t", j, " ", predicted[j]; 
        } 
}
