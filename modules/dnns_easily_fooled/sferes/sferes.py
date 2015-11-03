import sys, os
import subprocess
import commands

json_ok = True
try:
   import simplejson
except:
   json_ok = False
   print "WARNING simplejson not found some function may not work"

import glob
#import xml.etree.cElementTree as etree
import Options



def create_variants(bld, source, uselib_local, target, 
                    uselib, variants, includes=". ../../", 
                    cxxflags='',
                    json=''):
   # the basic one
   #   tgt = bld.new_task_gen('cxx', 'program')
   #   tgt.source = source
   #   tgt.includes = includes
   #   tgt.uselib_local = uselib_local
   #   tgt.uselib = uselib
   #   tgt.target = target
   # the variants
   c_src = bld.path.abspath() + '/'
   for v in variants:
      # create file
      suff = ''
      for d in v.split(' '): suff += d.lower() + '_'
      tmp = source.replace('.cpp', '')
      src_fname = tmp + '_' + suff[0:len(suff) - 1] + '.cpp'
      f = open(c_src + src_fname, 'w')
      f.write("// THIS IS A GENERATED FILE - DO NOT EDIT\n")
      for d in v.split(' '): f.write("#define " + d + "\n")
      f.write("#line 1 \"" + c_src + source + "\"\n")
      code = open(c_src + source, 'r')
      for line in code: f.write(line)
      bin_name = src_fname.replace('.cpp', '')
      bin_name = os.path.basename(bin_name)
      # create build
      tgt = bld.new_task_gen('cxx', 'program')
      tgt.source = src_fname
      tgt.includes = includes
      tgt.uselib_local = uselib_local
      tgt.uselib = uselib
      tgt.target = bin_name
      tgt.cxxflags = cxxflags


def create_exp(name):
   ws_tpl = """
#! /usr/bin/env python
def build(bld):
    obj = bld.new_task_gen('cxx', 'program')
    obj.source = '@exp.cpp'
    obj.includes = '. ../../'
    obj.uselib_local = 'sferes2'
    obj.uselib = ''
    obj.target = '@exp'
    obj.uselib_local = 'sferes2'
"""
   os.mkdir('exp/' + name)
   os.system("cp examples/ex_ea.cpp exp/" + name + "/" + name + ".cpp")
   wscript = open('exp/' + name + "/wscript", "w")
   wscript.write(ws_tpl.replace('@exp', name))
   

def parse_modules():
   if (not os.path.exists("modules.conf")):
      return []
   mod = open("modules.conf")
   modules = []
   for i in mod:
      if i[0] != '#' and len(i) != 1:
         modules += ['modules/' + i[0:len(i)-1]]
   return modules

def qsub(conf_file):
   tpl = """
#! /bin/sh
#? nom du job affiche
#PBS -N @exp
#PBS -o stdout
#PBS -b stderr
#PBS -M @email
# maximum execution time
#PBS -l walltime=@wall_time
# mail parameters
#PBS -m abe
# number of nodes
#PBS -l nodes=@nb_cores:ppn=@ppn
#PBS -l pmem=5200mb -l mem=5200mb
export LD_LIBRARY_PATH=@ld_lib_path
exec @exec
"""
   if os.environ.has_key('LD_LIBRARY_PATH'):
      ld_lib_path = os.environ['LD_LIBRARY_PATH']
   else:
      ld_lib_path = "''"
   home = os.environ['HOME']
   print 'LD_LIBRARY_PATH=' + ld_lib_path
    # parse conf
   conf = simplejson.load(open(conf_file))
   exps = conf['exps']
   nb_runs = conf['nb_runs']
   res_dir = conf['res_dir']
   bin_dir = conf['bin_dir']
   wall_time = conf['wall_time']
   use_mpi = "false"
   try: use_mpi = conf['use_mpi']
   except: use_mpi = "false"
   try: nb_cores = conf['nb_cores']
   except: nb_cores = 1
   try: args = conf['args']
   except: args = ''
   email = conf['email']
   if (use_mpi == "true"): 
      ppn = '1'
      mpirun = 'mpirun'
   else:
      nb_cores = 1; 
      ppn = '8'
      mpirun = ''
   
   for i in range(0, nb_runs):
      for e in exps:
         directory = res_dir + "/" + e + "/exp_" + str(i) 
         try:
            os.makedirs(directory)
         except:
            print "WARNING, dir:" + directory + " not be created"
         subprocess.call('cp ' + bin_dir + '/' + e + ' ' + directory, shell=True)
         fname = home + "/tmp/" + e + "_" + str(i) + ".job"
         f = open(fname, "w")
         f.write(tpl
                 .replace("@exp", e)
                 .replace("@email", email)
                 .replace("@ld_lib_path", ld_lib_path)
                 .replace("@wall_time", wall_time)
                 .replace("@dir", directory)
                 .replace("@nb_cores", str(nb_cores))
                 .replace("@ppn", ppn)
                 .replace("@exec", mpirun + ' ' + directory + '/' + e + ' ' + args))
         f.close()
         s = "qsub -d " + directory + " " + fname
         print "executing:" + s
         retcode = subprocess.call(s, shell=True, env=None)
         print "qsub returned:" + str(retcode)

def loadleveler(conf_file):
   tpl = """
# @ job_name=<name>
# @ output = $(job_name).$(jobid) 
# @ error = $(output) 
# @ job_type = serial 
# @ class = <class> 
# @ resources=ConsumableMemory(<memory>) ConsumableCpus(<cpu>)
# @ queue
export LD_LIBRARY_PATH=<ld_library_path>
cd <initial_dir>
./<exec>
"""
   if os.environ.has_key('LD_LIBRARY_PATH'):
      ld_lib_path = os.environ['LD_LIBRARY_PATH']
   else:
      ld_lib_path = "''"
   home = os.environ['HOME']
   print 'LD_LIBRARY_PATH=' + ld_lib_path
    # parse conf
   conf = simplejson.load(open(conf_file))
   jobname = conf['jobname']
   exps = conf['exps']
   nb_runs = conf['nb_runs']
   res_dir = conf['res_dir']
   bin_dir = conf['bin_dir']
   jobclass = conf['class']
   try:
      memory=conf['memory']
   except:
      memory=3000
   try:
      cpu=conf['cpu']
   except:
      cpu=1
   
   for i in range(0, nb_runs):
      for e in exps:
         directory = res_dir + "/" + e + "/exp_" + str(i) 
         try:
            os.makedirs(directory)
         except:
            print "WARNING, dir:" + directory + " cannot be created"
         subprocess.call('cp ' + bin_dir + '/' + e + ' ' + directory, shell=True)
         try:
            os.makedirs(home+"/tmp")
         except:
            pass
         fname = home + "/tmp/" + e + "_" + str(i) + ".job"
         f = open(fname, "w")
         f.write(tpl
                 .replace("<name>", jobname)
                 .replace("<ld_library_path>", ld_lib_path)
                 .replace("<class>", jobclass)
                 .replace("<initial_dir>", directory)
                 .replace("<memory>", str(memory))
                 .replace("<cpu>", str(cpu))
                 .replace("<exec>", e))
         f.close()
         s = "llsubmit "+ fname
         print "executing:" + s
         retcode = subprocess.call(s, shell=True, env=None)
         print "llsubmit returned:" + str(retcode)




def time_travel(conf_file):
   print 'time_travel, conf = ' + conf_file
   conf = simplejson.load(open(conf_file))
   dir = conf['dir']
   # get the diff
   patch = glob.glob(dir + '/*.diff')[0].split('/')
   patch = patch[len(patch) - 1]
   version = patch[0:len(patch) - len('.diff')]
   cwd = os.getcwd()
   patch = cwd + '/' + dir + '/' + patch
   # checkout
   print 'svn co -r ' + version
   os.system('cd ' + dir + ' && svn -r ' + version + ' co https://webia.lip6.fr:2004/svn/robur/sferes2')
   os.system('cd ' + dir + '/sferes2 && patch -p0 < '+ patch)
   os.chdir(cwd)


def get_gen(x):
   g1 = x.split('_')
   gen1 = int(g1[len(g1)-1])
   return gen1


def get_exe(conf_file):
    return os.path.split(conf_file.replace('.json', ''))[1]


def compare_gen(x, y):
    return get_gen(x) - get_gen(y)


def kill(conf_file):
   print 'kill, conf =' + conf_file
   exe = get_exe(conf_file)
   conf = simplejson.load(open(conf_file))
   machines = conf['machines']
   if conf['debug'] == 1:
      exe += '_debug'
   else:
      exe += '_opt'
   print 'kill '+ exe
   for m in machines:
       print m
       s = "ssh -o CheckHostIP=false -f " + m + \
           " killall -9 " + exe
       print s
       os.system(s)



def status(conf):
   # parse configuration
   print 'status, conf = ' + conf
   conf = simplejson.load(open(conf))
   exp = conf['exp']
   dir = conf['dir']

   exps = glob.glob(dir + '/exp_*/')
   total = 0.0
   for i in exps:
      glist = glob.glob(i + '*/gen_*')
      glist.sort(cmp=compare_gen)
      last = glist[len(glist) - 1]
      last_gen = get_gen(last)
      r = ''
      try:
         tree = etree.parse(last)
         l = tree.find("//x/_pareto_front/item/px/_fit/_objs")
         if l == None:
            l = tree.find("//x/_best/px/_fit/_value")
            total += float(l.text)
            r = l.text
         else:
            l = l[1:len(l)]
            total +=  float(l[0].text)
            for k in l:
               r +=  k.text + ' '
         print i + ' :\t' + str(last_gen) + '\t=> ' + r
      except:
        print "error"
   total /= len(exps)
   print "=> " + str(total)


def get_exp(conf_file):
   conf = simplejson.load(open(conf_file))
   return conf['exp'].sub('exp/', '')

def launch_exp(conf_file):
   print '--- launch exp ---'

   # parse configuration
   print 'launch, conf = ' + conf_file
   conf = simplejson.load(open(conf_file))
   machines = conf['machines']
   nb_runs = conf['nb_runs']
   exp = conf['exp']
   directory = conf['dir']
   debug = conf['debug']
   
   args = ""
   if 'args' in conf : args=conf['args']
   print 'exp = ' + exp
   print 'dir = ' + directory
   print 'nb_runs = ' + str(nb_runs)
   print 'debug = ' + str(debug)
   print 'machines =' + str(machines)
   print 'args =' + str(args)

   # copy binaries (debug and opt) & json file
   exe = get_exe(conf_file)
   try:
      os.makedirs(directory + '/bin')
      os.system('cp ' + 'build/default/' + exp +'/'+exe+ ' ' + directory + '/bin/' + exe + '_opt')
      os.system('cp ' + 'build/debug/' + exp +'/'+exe+ ' ' + directory + '/bin/' + exe + '_debug')
      print conf
      print directory
      os.system('cp ' + conf_file + ' ' + directory)
      # create directories
      for i in range(0, nb_runs * len(machines)):
         os.makedirs(directory + '/exp_' + str(i))
   except:
      print '/!\ files exist, I cannot replace them'
      return
   print 'dirs created'

   # make a svn diff
   status, version = commands.getstatusoutput('svnversion') 
   if version[len(version)-1] == 'M':
      version = version[0:len(version)-1]
   os.system('svn diff >' + directory + '/' + version + '.diff')
   print 'diff done [version=' + version + ']'

   # run on each machines
   if debug == 1:
      exe = exe + '_debug'
   else:
      exe = exe + '_opt'
   if os.environ.has_key('LD_LIBRARY_PATH'):
      ld_lib_path = os.environ['LD_LIBRARY_PATH']
   else:
      ld_lib_path = "''"
   k = 0
   pids = []
   for m in machines.iterkeys() :
      pid = os.fork() 
      if (pid == 0): #son
         for i in range(0, machines[m]):
            if m == 'localhost':
               s = "export LD_LIBRARY_PATH=" + ld_lib_path + \
                   " && cd " + os.getcwd() + '/' + directory + '/exp_'+ str(i + k) + \
                   " && " +  os.getcwd() + '/' + directory + '/bin/' + exe + " " + args + \
                   " 1> stdout 2> stderr"
            else:
               s = "ssh -o CheckHostIP=false " + m + \
                   " 'export LD_LIBRARY_PATH=" + ld_lib_path + \
                   " && cd " + os.getcwd() + '/' + directory + '/exp_'+ str(i + k) + \
                   " && " +  os.getcwd() + '/' + directory + '/bin/' + exe + " " + args + \
                   " 1> stdout 2> stderr'"
            print 'run ' + str(i + k) + ' on ' + m
	    print s
            ret = subprocess.call(s, shell=True)            
            print "ret = " + str(ret)
         exit(0)
      pids += [pid]
      k += machines[m]
   print "waitpid..."
   for i in pids:
      os.waitpid(i, 0)
