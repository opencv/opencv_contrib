import sys
if sys.version_info >= (3, 10):
    import collections.abc
    IterableType = collections.abc.Iterable
else:
    import collections
    IterableType = collections.Iterable
from textwrap import fill
from filters import *
try:
  # Python 2.7+
  basestring
except NameError:
  # Python 3.3+
  basestring = str


valid_types = (
    'int', 'bool', 'float', 'double', 'size_t', 'char',
    'Mat', 'Scalar', 'String',
    'TermCriteria', 'Size', 'Point', 'Point2f', 'Point2d', 'Rect', 'RotatedRect',
    'RNG', 'DMatch', 'Moments',
    'vector_Mat', 'vector_Point', 'vector_int', 'vector_float', 'vector_double', 'vector_String', 'vector_uchar', 'vector_Rect', 'vector_DMatch', 'vector_KeyPoint',
    'vector_Point2f', 'vector_vector_char', 'vector_vector_DMatch', 'vector_vector_KeyPoint',
    'Ptr_StereoBM', 'Ptr_StereoSGBM', 'Ptr_FeatureDetector', 'Ptr_CLAHE', 'Ptr_LineSegmentDetector', 'Ptr_AlignMTB', 'Ptr_CalibrateDebevec',
    'Ptr_CalibrateRobertson', 'Ptr_DenseOpticalFlow', 'Ptr_DualTVL1OpticalFlow', 'Ptr_MergeDebevec', 'Ptr_MergeMertens', 'Ptr_MergeRobertson',
    'Ptr_Stitcher', 'Ptr_Tonemap', 'Ptr_TonemapDrago', 'Ptr_TonemapDurand', 'Ptr_TonemapMantiuk', 'Ptr_TonemapReinhard', 'Ptr_float',
    # Not supported:
    #vector_vector_KeyPoint
)

class ParseTree(object):
    """
    The ParseTree class produces a semantic tree of C++ definitions given
    the output of the CppHeaderParser (from opencv/modules/python/src2/hdr_parser.py)

    The full hierarchy is as follows:

      Namespaces
        |
        |- name
        |- Classes
            |
            |- name
            |- Methods
            |- Constants
        |- Methods
            |
            |- name
            |- static (T/F)
            |- return type
            |- required Arguments
                |
                |- name
                |- const (T/F)
                |- reference ('&'/'*')
                |- type
                |- input
                |- output (pass return by reference)
                |- default value
            |- optional Arguments
        |- Constants
            |
            |- name
            |- const (T/F)
            |- reference ('&'/'*')
            |- type
            |- value

    The semantic tree contains substantial information for easily introspecting
    information about objects. How many methods does the 'core' namespace have?
    Does the 'randn' method have any return by reference (output) arguments?
    How many required and optional arguments does the 'add' method have? Is the
    variable passed by reference or raw pointer?

    Individual definitions from the parse tree (Classes, Functions, Constants)
    are passed to the Jinja2 template engine where they are manipulated to
    produce Matlab mex sources.

    A common call tree for constructing and using a ParseTree object is:

      # parse a set of definitions into a dictionary of namespaces
      parser = CppHeaderParser()
      ns['core'] = parser.parse('path/to/opencv/core.hpp')

      # refactor into a semantic tree
      parse_tree = ParseTree()
      parse_tree.build(ns)

      # iterate over the tree
      for namespace in parse_tree.namespaces:
        for clss in namespace.classes:
          # do stuff
        for method in namespace.methods:
          # do stuff

    Calling 'print' on a ParseTree object will reconstruct the definitions
    to produce an output resembling the original C++ code.
    """
    def __init__(self, namespaces=None):
        self.namespaces = namespaces if namespaces else []

    def __str__(self):
        return '\n\n\n'.join(ns.__str__() for ns in self.namespaces)

    def build(self, namespaces):
        babel = Translator()
        for name, definitions in namespaces.items():
            class_tree = {}
            methods = []
            constants = []
            for defn in definitions:
                try:
                    obj = babel.translate(defn)
                except Exception as e:
                    print(e)
                    obj = None
                if obj is None:
                    continue
                if type(obj) is Class or obj.clss:
                    self.insertIntoClassTree(obj, class_tree)
                elif type(obj) is Method:
                    methods.append(obj)
                elif type(obj) is Constant:
                    constants.append(obj)
                else:
                    raise TypeError('Unexpected object type: '+str(type(obj)))
            self.namespaces.append(Namespace(name, constants, list(class_tree.values()), methods))

    def insertIntoClassTree(self, obj, class_tree):
        cname = obj.name if type(obj) is Class else obj.clss
        if not cname:
            return
        if not cname in class_tree:
          # add a new class to the tree
            class_tree[cname] = Class(cname)
        # insert the definition into the class
        val = class_tree[cname]
        if type(obj) is Method:
            val.methods.append(obj)
        elif type(obj) is Constant:
            val.constants.append(obj)
        else:
            raise TypeError('Unexpected object type: '+str(type(obj)))



class Translator(object):
    """
    The Translator class does the heavy lifting of translating the nested
    list representation of the hdr_parser into individual definitions that
    are inserted into the ParseTree.
    Translator consists of a top-level method: translate()
    along with a number of helper methods: translateClass(), translateMethod(),
    translateArgument(), translateConstant(), translateName(), and
    translateClassName()
    """
    def translate(self, defn):
        # --- class ---
        # classes have 'class' prefixed on their name
        if 'class' in defn[0].split(' ') or 'struct' in defn[0].split(' '):
            return self.translateClass(defn)
        # --- operators! ---
        #TODO: implement operators: http://www.mathworks.com.au/help/matlab/matlab_oop/implementing-operators-for-your-class.html
        if 'operator' in defn[0]:
            return
        # --- constant ---
        elif convertibleToInt(defn[1]):
            return self.translateConstant(defn)
        # --- function ---
        # functions either need to have input arguments, or not uppercase names
        elif defn[3] or not self.translateName(defn[0]).split('_')[0].isupper():
            return self.translateMethod(defn)
        # --- constant ---
        else:
            return self.translateConstant(defn)

    def translateClass(self, defn):
        return Class()

    def translateMethod(self, defn, class_tree=None):
        name = self.translateName(defn[0])
        clss = self.translateClassName(defn[0])
        rtp  = defn[1]
        static = True if 'S' in ''.join(defn[2]) else False
        args = defn[3]
        req  = []
        opt = []
        for arg in args:
            if arg:
                a = self.translateArgument(arg)
                opt.append(a) if a.default else req.append(a)
        return Method(name, clss, static, '', rtp, False, req, opt)

    def translateConstant(self, defn):
        const = True if 'const' in defn[0] else False
        name  = self.translateName(defn[0])
        clss  = self.translateClassName(defn[0])
        tp    = 'int'
        val   = defn[1]
        return Constant(name, clss, tp, const, '', val)

    def translateArgument(self, defn):
        modifiers = defn[3]
        ref   = '*' if '*' in defn[0] else ''
        ref   = '&' if '&' in defn[0] or '/Ref' in modifiers else ref
        const = '/C' in modifiers
        tp    = " ".join([word for word in defn[0].replace(ref, '').split() if not ' const ' in ' '+word+' '])
        name = defn[1]
        default = defn[2] if defn[2] else ''
        I = True if '/I' in modifiers or not '/O' in modifiers else False
        O = True if '/O' in modifiers else False
        return Argument(name, tp, const, I, O, ref, default)

    def translateName(self, name):
        return name.split(' ')[-1].split('.')[-1]

    def translateClassName(self, name):
        name  = name.split(' ')[-1]
        parts = name.split('.')
        return parts[-2] if len(parts) > 1 and not parts[-2] == 'cv' else ''



class Namespace(object):
    """
    Namespace
      |
      |- name
      |- Constants
      |- Methods
      |- Constants
    """
    def __init__(self, name='', constants=None, classes=None, methods=None):
        self.name = name
        self.constants = constants if constants else []
        self.classes   = classes   if classes   else []
        self.methods = methods if methods else []

    def __str__(self):
        return 'namespace '+self.name+' {\n\n'+\
          ('\n'.join(c.__str__() for c in self.constants)+'\n\n' if self.constants else '')+\
          ('\n'.join(f.__str__() for f in self.methods)+'\n\n'   if self.methods   else '')+\
          ('\n\n'.join(o.__str__() for o in self.classes)        if self.classes   else '')+'\n};'

class Class(object):
    """
    Class
      |
      |- name
      |- Methods
      |- Constants
    """
    def __init__(self, name='', namespace='', constants=None, methods=None):
        self.name = name
        self.namespace = namespace
        self.constants = constants if constants else []
        self.methods = methods if methods else []

    def __str__(self):
        return 'class '+self.name+' {\n\t'+\
          ('\n\t'.join(c.__str__() for c in self.constants)+'\n\n\t' if self.constants else '')+\
          ('\n\t'.join(f.__str__() for f in self.methods)            if self.methods   else '')+'\n};'

class Method(object):
    """
    Method
    int VideoWriter::read( cv::Mat& frame, const cv::Mat& mask=cv::Mat() );
    ---    -----     ----     --------           ----------------
    rtp    class     name     required               optional

    name      the method name
    clss      the class the method belongs to ('' if free)
    static    static?
    namespace the namespace the method belongs to ('' if free)
    rtp       the return type
    const     const?
    req       list of required arguments
    opt       list of optional arguments
    """
    def __init__(self, name='', clss='', static=False, namespace='', rtp='', const=False, req=None, opt=None):
        self.name  = name
        self.clss  = clss
        self.constructor = True if name == clss else False
        self.static = static
        self.const = const
        self.namespace = namespace
        self.rtp = rtp
        self.req = req if req else []
        self.opt = opt if opt else []

    def __str__(self):
        return (self.rtp+' ' if self.rtp else '')+self.name+'('+\
          ', '.join(arg.__str__() for arg in self.req+self.opt)+\
          ')'+(' const' if self.const else '')+';'

class Argument(object):
    """
    Argument
    const cv::Mat&  mask=cv::Mat()
    -----  ---- --- ----  -------
    const   tp  ref name  default

    name    the argument name
    tp      the argument type
    const   const?
    I       is the argument treated as an input?
    O       is the argument treated as an output (return by reference)
    ref     is the argument passed by reference? ('*'/'&')
    default the default value of the argument ('' if required)
    """
    def __init__(self, name='', tp='', const=False, I=True, O=False, ref='', default=''):
        self.name = name
        self.tp   = tp
        self.ref  = ref
        self.I    = I
        self.O    = O
        self.const = const
        self.default = default
        if not tp in valid_types:
            raise Exception("Non-supported argument type: {} (name: {})".format(tp, name))

    def __str__(self):
        return ('const ' if self.const else '')+self.tp+self.ref+\
                ' '+self.name+('='+self.default if self.default else '')

class Constant(object):
    """
    Constant
    DFT_COMPLEX_OUTPUT = 12;
         ----          -------
         name          default

    name    the name of the constant
    clss    the class that the constant belongs to ('' if free)
    tp      the type of the constant ('' if int)
    const   const?
    ref     is the constant a reference? ('*'/'&')
    default default value, required for constants
    """
    def __init__(self, name='', clss='', tp='', const=False, ref='', default=''):
        self.name = name
        self.clss = clss
        self.tp   = tp
        self.ref  = ref
        self.const = const
        self.default = default

    def __str__(self):
        return ('const ' if self.const else '')+self.tp+self.ref+\
                ' '+self.name+('='+self.default if self.default else '')+';'

def constants(tree):
    """
    recursive generator to strip all Constant objects from the ParseTree
    and place them into a flat dictionary of { name, value (default) }
    """
    if isinstance(tree, dict) and 'constants' in tree and isinstance(tree['constants'], list):
        for node in tree['constants']:
            yield (node['name'], node['default'])
    if isinstance(tree, dict):
        for key, val in tree.items():
            for gen in constants(val):
                yield gen
    if isinstance(tree, list):
        for val in tree:
            for gen in constants(val):
                yield gen


def todict(obj):
    """
    Recursively convert a Python object graph to sequences (lists)
    and mappings (dicts) of primitives (bool, int, float, string, ...)
    """
    if isinstance(obj, basestring):
        return obj
    elif isinstance(obj, dict):
        return dict((key, todict(val)) for key, val in obj.items())
    elif isinstance(obj, IterableType):
        return [todict(val) for val in obj]
    elif hasattr(obj, '__dict__'):
        return todict(vars(obj))
    elif hasattr(obj, '__slots__'):
        return todict(dict((name, getattr(obj, name)) for name in getattr(obj, '__slots__')))
    return obj
