#!/usr/bin/python
import os, commands, sys

projectGroup = "GroupName"
numRuns = 1
runName = "Run"
yourMail = "anguyen8@uwyo.edu"
runName = runName+'_'
numCores = 128 # A multiple of 16
numNodes = str(numCores/16)

pathCurrentDir = os.path.dirname(os.path.abspath(__file__)) # Path to current directory without trailing slash '/'
executable = pathCurrentDir + "/build/default/exp/images/images "

YourInputParameter1 = str(10)
#options = ""
#options = "--your_option your_value --your_second_option $your_variable --your_third_variable " + YourInputParameter1
options = "$seed_num "
scriptFileName = "launchScript.sh"


def printScriptFile():
    scriptFile  = open(scriptFileName,'w',)

    #This will print the header of the launch script
    scriptFile.write( "#!/bin/bash\n")
    scriptFile.write( "\n")
    scriptFile.write( "#Do not edit. File automatically generated\n")
    scriptFile.write( "\n")

    #Write any modules, environment variables or other commands your program needs before it is being executed here
    #Always load compiler module first. See command "module spider" for available modules
    #scriptFile.write( "module load intel/14.0.0\n")
    scriptFile.write( "module load gnu/4.8.2\n")
    #scriptFile.write( "module load tbb/4.2.2\n")
    scriptFile.write( "module load cuda/5.5\n")
    scriptFile.write( "module load openmpi/1.6.5\n")
    #scriptFile.write( "module load allinea\n")

    #Here we change to the directory where the experiment will be executed
    #Note that experiment dir is a variable that is not defined here
    scriptFile.write( "echo \"Changing to directory: \" $experimentDir\n")
    scriptFile.write( "cd $experimentDir\n")
    scriptFile.write( "\n")


    #scriptFile.write( "ln -s ../config/voxelize\n")

    #Echo what we will execute to a file called runToBe
    scriptFile.write( "echo \" " + executable + options + " > thisistheoutput 2> err.log\" > runToBe\n")

    #Actually execute your program
    #scriptFile.write( "time " + executable + options + " > thisistheoutput 2> err.log\n")
    scriptFile.write( "time mpirun --mca mpi_leave_pinned 0 --mca mpi_warn_on_fork 0 -np " + str(numCores) + " " + executable + options + " > thisistheoutput 2> err.log\n")

    #This will print the footer of the launch script
    scriptFile.write( "\n")
    scriptFile.write( "echo \"Done with run\"\n")
    scriptFile.close()

    #Here we make the launch script executable
    os.system("chmod +x " + scriptFileName)

################
# main starts here #
################

print 'Starting a batch of runs called: ' + runName

printScriptFile()

for i in range(0,numRuns):
    #i += 1
    #Create a new directory for our run
    runNumStr = str(i)
    runDirShort = "run_" + runNumStr.zfill(numRuns/10)
    # If there is a path.. continue
    if os.path.isdir(runDirShort):
        print runDirShort + " already exists. Abort!"
        #sys.exit(3)
    else:	# Create a new run_x folder
        command = "mkdir "  + runDirShort
        os.system(command)

    #Create the string to our new directory
    pwd = commands.getoutput("pwd")
    experimentDir = pwd + "/" + runDirShort

    #Set your own variables
    variableThatShouldBeDifferentForEachRun = str(i) 

    #Create the command that will submit your experiment
    command = ("qsub -v seed_num=" + variableThatShouldBeDifferentForEachRun
               + ",experimentDir=" + experimentDir 
               + " -m abe"
               + " -l walltime=05:00:00:00"
	      #+ " -l mem=64gb"
	       + " -l nodes="+ numNodes +":ppn=16"
	      #+ " -l nodes=1:ppn=16:gpus=2:gpu"
               + " -M " + yourMail
               + " -A " + projectGroup
               + " -o " + experimentDir + "/myOut"
               + " -e " + experimentDir + "/myErr"
               + " -N " + runName + str(i) + " " + scriptFileName)
    print command

    #Note: qsub variable list is a comma-separated list without spaces between variables. 
    #Ex: qsub -v var1=a,var2=b

    #Launch your experiment
    os.system(command)
