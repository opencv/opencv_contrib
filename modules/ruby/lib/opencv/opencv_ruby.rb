module OpenCV
    def self.included(obj)
        obj.class_eval do
            def cv
                Cv
            end
            def std
                Std
            end
        end
    end

    def self.std
        Std
    end

    def self.cv
        Cv
    end

    module Std
        class Vector
            class Cv_Mat
                def self.to_native(obj,context)
                    if obj.is_a?(Vector::Std_Vector_Cv_Point2f)
                        t = Vector::Cv_Mat.new
                        obj.each do |e|
                            t << Cv::Mat.new(e.size,2,CV_32FC1,e.data,Cv::Mat::AUTO_STEP)
                        end
                        t.__obj_ptr__
                    elsif obj.is_a?(Vector::Std_Vector_Cv_Point2d)
                        t = Vector::Cv_Mat.new
                        obj.each do |e|
                            t << Cv::Mat.new(e.size,2,CV_64FC1,e.data,Cv::Mat::AUTO_STEP)
                        end
                        t.__obj_ptr__
                    elsif obj.is_a?(Vector::Std_Vector_Cv_Point)
                        t = Vector::Cv_Mat.new
                        obj.each do |e|
                            t << Cv::Mat.new(e.size,2,CV_32SC1,e.data,Cv::Mat::AUTO_STEP)
                        end
                        t.__obj_ptr__
                    else
                        rbind_to_native(obj,context)
                    end
                end
            end
            Fixnum = Int
        end
    end
    include Std

    module Cv
        # reflect some typedefs
        GoodFeatureToTrackDetector = GFTTDetector if defined? GFTTDetector
        StarFeatureDetector = StarDetector if defined? StarDetector

        def self.min_max_loc(src,min_loc = Point.new,max_loc = Point.new,mask = Mat.new)
            p = FFI::MemoryPointer.new(:double,2)
            Rbind::cv_min_max_loc(src, p[0], p[1], min_loc, max_loc, mask)
            [p[0].read_double,p[1].read_double]
        end

        class Size
            def *(val)
                Size.new(width*val,height*val)
            end
            def +(val)
                Size.new(width+val,height+val)
            end
            def -(val)
                Size.new(width-val,height-val)
            end
        end
        class String
            def self.to_native(obj,context)
                if obj.is_a? ::String
                    str = obj.to_str
                    OpenCV::Cv::String.new(str,str.length).__obj_ptr__
                else
                    rbind_to_native(obj,context)
                end
            end
            def to_s
                c_str
            end
        end

        class Point
            def self.to_native(obj,context)
                if obj.is_a? ::OpenCV::Cv::Point2f
                    OpenCV::Cv::Point.new(obj.x,obj.y).__obj_ptr__
                else
                    rbind_to_native(obj,context)
                end
            end
        end

        module Vec2x
            def self.included base
                base.instance_eval do
                    def to_native(obj,context)
                        if obj.is_a? ::OpenCV::Cv::Point
                            self.new(obj.x,obj.y).__obj_ptr__
                        elsif obj.is_a? ::OpenCV::Cv::Point2f
                            self.new(obj.x,obj.y).__obj_ptr__
                        elsif obj.is_a? ::OpenCV::Cv::Point2d
                            self.new(obj.x,obj.y).__obj_ptr__
                        elsif obj.is_a?(::OpenCV::Cv::Mat) && obj.rows == 2 && obj.cols == 1
                            self.new(obj[0],obj[1]).__obj_ptr__
                        else
                            rbind_to_native(obj,context)
                        end
                    end
                end
            end
        end

        module Vec3x
            def self.included base
                base.instance_eval do
                    def to_native(obj,context)
                        if obj.is_a? ::OpenCV::Cv::Point3f
                            self.new(obj.x,obj.y,obj.z).__obj_ptr__
                        elsif obj.is_a? ::OpenCV::Cv::Point3d
                            self.new(obj.x,obj.y,obj.z).__obj_ptr__
                        elsif obj.is_a?(::OpenCV::Cv::Mat) && obj.rows == 3 && obj.cols == 1
                            self.new(obj[0],obj[1],obj[2]).__obj_ptr__
                        else
                            rbind_to_native(obj,context)
                        end
                    end
                end
            end
        end

        module Vecxd
            def [](i)
                raise "Out of bound #{i}" if i < 0 || i >= self.class::SIZE
                val.get_float64(i*8)
            end
            def []=(i,val0)
                raise "Out of bound #{i}" if i < 0 || i >= self.class::SIZE
                val.put_float64(i*8,val0)
            end
            def to_a
                val.get_array_of_float64(0,self.class::SIZE)
            end
        end

        module Vecxf
            def [](i)
                raise "Out of bound #{i}" if i < 0 || i >= self.class::SIZE
                val.get_float32(i*4)
            end
            def []=(i,val0)
                raise "Out of bound #{i}" if i < 0 || i >= self.class::SIZE
                val.put_float32(i*4,val0)
            end
            def to_a
                val.get_array_of_float32(0,self.class::SIZE)
            end
        end

        module Vecxi
            def [](i)
                raise "Out of bound #{i}" if i < 0 || i >= self.class::SIZE
                val.get_int(i*FFI.type_size(FFI::Type::INT))
            end
            def []=(i,val0)
                raise "Out of bound #{i}" if i < 0 || i >= self.class::SIZE
                val.put_int(i*FFI.type_size(FFI::Type::INT),val0)
            end
            def to_a
                val.get_array_of_int(0,self.class::SIZE)
            end
        end

        class Vec2d;include Vecxd;include Vec2x; SIZE=2;end
        class Vec2f;include Vecxf;include Vec2x; SIZE=2;end
        class Vec2i;include Vecxi;include Vec2x; SIZE=2;end

        class Vec3d;include Vecxd;include Vec3x; SIZE=3;end
        class Vec3f;include Vecxf;include Vec3x; SIZE=3;end
        class Vec3i;include Vecxi;include Vec3x; SIZE=3;end

        class Vec4d;include Vecxd; SIZE=4;end
        class Vec4f;include Vecxf; SIZE=4;end
        class Vec4i;include Vecxi; SIZE=4;end

        class Vec6d;include Vecxd; SIZE=6;end
        class Vec6f;include Vecxf; SIZE=6;end
        class Vec6i;include Vecxi; SIZE=6;end

        class Scalar; include Vecxd; SIZE=4;end

        class FileStorage
            include Enumerable

            def each(&block)
                if block_given?
                    root.each(&block)
                else
                    to_enum(:each)
                end
            end

            def <<(val)
                if val.is_a?(Fixnum)
                    write_int(val)
                elsif val.is_a?(Float)
                    write_double(val)
                else
                    name = val.class.name.split("::").last.downcase
                    send("write_#{name}",val)
                end
            end

            def method_missing(m,*args)
                if args.empty?
                    self[m.to_s]
                else
                    super
                end
            end
        end

        class FileNode
            include Enumerable
            alias :empty? :empty
            alias :int? :isInt
            alias :real? :isReal
            alias :string? :isString
            alias :map? :isMap
            alias :seq? :isSeq

            def each(&block)
                if block_given?
                    iter = self.begin()
                    while(iter != self.end())
                        yield(iter.to_node)
                        iter.plusplus_operator
                    end
                else
                    to_enum(:each)
                end
            end

            def to_array_of_int
                raise RuntimeError, "FileNode is not the root node of a sequence" unless seq?
                map(&:to_int)
            end

            def to_array_of_float
                raise RuntimeError, "FileNode is not the root node of a sequence" unless seq?
                map(&:to_float)
            end

            def to_array_of_double
                raise RuntimeError, "FileNode is not the root node of a sequence" unless seq?
                map(&:to_double)
            end

            def to_array_of_string
                raise RuntimeError, "FileNode is not the root node of a sequence" unless seq?
                map(&:to_string)
            end

            def to_mat
                raise RuntimeError, "FileNode is empty" if empty?
                raise RuntimeError, "FileNode is not storing a Mat" unless isMap
                val = Cv::Mat.new
                read_mat(val)
                val
            end

            def to_float
                raise RuntimeError, "FileNode is empty" if empty?
                raise RuntimeError, "FileNode is not storing a float" unless isReal
                p = FFI::MemoryPointer.new(:float,1)
                read_float(p)
                p.get_float32 0
            end

            def to_double
                raise RuntimeError, "FileNode is empty" if empty?
                raise RuntimeError, "FileNode is not storing a double" unless isReal
                p = FFI::MemoryPointer.new(:uchar,8)
                read_double(p)
                p.get_float64 0
            end

            def to_int
                raise RuntimeError, "FileNode is empty" if empty?
                raise RuntimeError, "FileNode is not storing a double" unless isInt
                p = FFI::MemoryPointer.new(:int,1)
                read_int(p)
                p.get_int32 0
            end

            def to_string
                raise RuntimeError, "FileNode is empty" if empty?
                raise RuntimeError, "FileNode is not storing a string" unless isString
                str = Cv::String.new
                read_string(str)
                str
            end

            def method_missing(m,*args)
                if args.empty? && map?
                    self[m.to_s]
                else
                    super
                end
            end
        end

        class Mat
            include Enumerable
            DISPLAYED_ROWS_MAX = 100
            DISPLAYED_COLS_MAX = 100

            class << self
                alias :rbind_new :new

                def new(*args)
                    # allow Mat.new([123,23],[2332,32])
                    if !args.find{|a| !a.is_a?(Array)} && args.size() > 1
                        rbind_new(args)
                    else
                        rbind_new(*args)
                    end
                end
            end

            def self.to_native(obj,context)
                if obj.is_a?(Std::Vector::Cv_Point)
                    Cv::Mat.new(obj.size,1,CV_32SC2,obj.data,Cv::Mat::AUTO_STEP).__obj_ptr__
                elsif obj.is_a?(Std::Vector::Cv_Point2f)
                    Cv::Mat.new(obj.size,1,CV_32FC2,obj.data,Cv::Mat::AUTO_STEP).__obj_ptr__
                elsif obj.is_a?(Std::Vector::Cv_Point2d)
                    Cv::Mat.new(obj.size,1,CV_64FC2,obj.data,Cv::Mat::AUTO_STEP).__obj_ptr__
                elsif obj.is_a?(Std::Vector::Cv_Point3f)
                    Cv::Mat.new(obj.size,1,CV_32FC3,obj.data,Cv::Mat::AUTO_STEP).__obj_ptr__
                elsif obj.is_a?(Std::Vector::Cv_Point3d)
                    Cv::Mat.new(obj.size,1,CV_64FC3,obj.data,Cv::Mat::AUTO_STEP).__obj_ptr__
                elsif obj.is_a?(Std::Vector::Int)
                    Cv::Mat.new(obj.size,1,CV_32SC1,obj.data,Cv::Mat::AUTO_STEP).__obj_ptr__
                elsif obj.is_a?(Std::Vector::Float)
                    Cv::Mat.new(obj.size,1,CV_32FC1,obj.data,Cv::Mat::AUTO_STEP).__obj_ptr__
                elsif obj.is_a?(Std::Vector::Double)
                    Cv::Mat.new(obj.size,1,CV_64FC1,obj.data,Cv::Mat::AUTO_STEP).__obj_ptr__
                elsif obj.is_a?(Array)
                    h,w,e= if obj.first.is_a? Array
                               if obj.find {|array| array.find(Float)}
                                   [obj.size,obj.first.size,obj.first.first.to_f]
                               else
                                   [obj.size,obj.first.size,obj.first.first]
                               end
                           else
                               if obj.find(Float)
                                   [obj.size,1,obj.first.to_f]
                               else
                                   [obj.size,1,obj.first]
                               end
                           end
                    setter,step,type = if e.is_a? Fixnum
                                           [:put_array_of_int32,4*w,CV_32SC1]
                                       elsif e.is_a? Float
                                           [:put_array_of_float64,8*w,CV_64FC1]
                                       else
                                           raise ArgumentError,"cannot connvert array of #{e.class} to Cv::Mat"
                                       end
                    mat = Mat.new(h,w,type)
                    ptr = mat.data
                    setter = ptr.method(setter)
                    if h == 1 || w == 1
                        setter.call(0,obj.flatten)
                    else
                        obj.each_with_index do |row,i|
                            raise ArgumentError, "number of row elements must be equal for each row" if row.size != w
                            setter.call(i*step,row)
                        end
                    end
                    mat.__obj_ptr__
                else
                    rbind_to_native(obj,context)
                end
            end

            def -@
                Cv::Mat.zeros(rows,cols,type)-self
            end

            def at(i,j=nil,k=0)
                raise ArgumentError,"channel #{k} out of bound" if k >= channels
                i,j = if j == nil
                          if i.is_a?(Cv::Point)
                              [i.y,i.x]
                          elsif rows == 1
                              [0,i]
                          else
                              [i,0]
                          end
                      else
                          [i,j]
                      end
                if i >= rows || i < 0 || j >= cols || j <0
                    raise ArgumentError,"out of bound #{i}/#{j} #{rows}/#{cols}"
                end
                j = j*channels+k
                case type & 7
                when CV_8U
                    data.get_uint8(i*step+j)
                when CV_16U
                    data.get_uint16(i*step+j*2)
                when CV_16S
                    data.get_int16(i*step+j*2)
                when CV_32S
                    data.get_int32(i*step+j*4)
                when CV_32F
                    data.get_float32(i*step+j*4)
                when CV_64F
                    data.get_float64(i*step+j*8)
                else
                    raise "cannot connvert #{self.class} of type #{type} to ruby"
                end
            end

            def coerce(other)
                case other
                when Float
                    [other,to_f]
                when Fixnum 
                    [other,to_i]
                else
                    raise TypeError, "#{self.class} can't be coerced into #{other.class}"
                end
            end

            def to_f
                if rows == 1 && cols == 1
                    at(0,0).to_f
                else
                    raise "Matrix #{self} has more than one element"
                end
            end

            def to_i
                if rows == 1 && cols == 1
                    at(0,0).to_i
                else
                    raise "Matrix #{self} has more than one element"
                end
            end

            def set(i,j,k=nil,val=nil)
                k,val = if val == nil
                            [val,k]
                        else
                            [k,val]
                        end
                k ||= 0
                raise ArgumentError,"channel #{k} out of bound" if k >= channels
                i,j,val = if val == nil
                              if i.is_a?(Cv::Point)
                                  [i.y,i.x,j]
                              elsif rows == 1
                                  [0,i,j]
                              else
                                  [i,0,j]
                              end
                          else
                              [i,j,val]
                          end
                if i >= rows || i < 0 || j >= cols || j <0
                    raise ArgumentError,"out of bound #{i}/#{j}"
                end
                j = j*channels+k
                case type & 7
                when CV_8U
                    data.put_uint8(i*step+j,val)
                when CV_16U
                    data.put_uint16(i*step+j*2,val)
                when CV_16S
                    data.put_int16(i*step+j*2,val)
                when CV_32S
                    data.put_int32(i*step+j*4,val)
                when CV_32F
                    data.put_float32(i*step+j*4,val)
                when CV_64F
                    data.put_float64(i*step+j*8,val)
                else
                    raise "cannot connvert #{self.class} of type #{type} to ruby"
                end
            end

            def [](i,j=nil,k=0)
                at(i,j,k)
            end

            def []=(i,j,k=nil,val=nil)
                set(i,j,k,val)
            end

            def -(val)
                if val.is_a? Float
                    Rbind::cv_mat_operator_minus2( self, val)
                elsif val.is_a? Fixnum
                    Rbind::cv_mat_operator_minus3( self, val)
                else
                    Rbind::cv_mat_operator_minus( self, val)
                end
            end

            def +(val)
                if val.is_a? Float
                    Rbind::cv_mat_operator_plus2( self, val)
                elsif val.is_a? Fixnum
                    Rbind::cv_mat_operator_plus3( self, val)
                else
                    Rbind::cv_mat_operator_plus( self, val)
                end
            end

            def /(val)
                if val.is_a? Float
                    Rbind::cv_mat_operator_div2( self, val)
                elsif val.is_a? Fixnum
                    Rbind::cv_mat_operator_div3( self, val)
                else
                    Rbind::cv_mat_operator_div( self, val)
                end
            end

            def *(val)
                if val.is_a? Float
                    Rbind::cv_mat_operator_mult2( self, val)
                elsif val.is_a? Fixnum
                    Rbind::cv_mat_operator_mult3( self, val)
                else
                    Rbind::cv_mat_operator_mult( self, val)
                end
            end

            def pretty_print(pp)
                if(rows <= DISPLAYED_ROWS_MAX && cols <= DISPLAYED_COLS_MAX)
                    format = case type & 7
                             when CV_8U
                                 '%3.u'
                             else
                                 '%6.3f'
                             end
                    str = to_a.map do |r|
                        str = r.map do |e|
                            sprintf(format,e)
                        end.join(" ")
                        "|#{str}|"
                    end.join("\n")
                        pp.text str
                else
                    pp.text self.to_s
                end
            end

            def each
                if block_given?
                    0.upto(rows-1) do |row|
                        0.upto(cols-1) do |col|
                            yield at(row,col)
                        end
                    end
                else
                    to_enum(:each)
                end
            end

            def each_row_with_index(&block)
                if block_given?
                    r = rows
                    0.upto(r-1) do |i|
                        yield(row(i),i)
                    end
                else
                    to_enum(:each_row_with_index)
                end
            end

            def each_col_with_index(&block)
                if block_given?
                    c = cols
                    0.upto(c-1) do |i|
                        yield(col(i),i)
                    end
                else
                    to_enum(:each_col_with_index)
                end
            end

            def each_row(&block)
                if block_given?
                    each_row_with_index do |r,i|
                        yield(r)
                    end
                else
                    to_enum(:each_row)
                end
            end

            def each_col(&block)
                if block_given?
                    each_col_with_index do |c,i|
                        yield(c)
                    end
                else
                    to_enum(:each_col)
                end
            end

            # returns a string compatible to matlab's MAT-file
            def to_MAT(variable_name)
<<eos
% Created by ropencv, #{Time.now}
% name: #{variable_name}
% type: matrix
% rows: #{rows}
% columns: #{cols}
#{to_a.map{|row|row.join(" ")}.join("\n")}
eos
            end

            def to_a
                h,w,c,s,ptr = [rows,cols,channels,step,data]
                getter = case type & 7
                         when CV_8U
                             ptr.method(:get_array_of_uint8)
                         when CV_16U
                             ptr.method(:get_array_of_uint16)
                         when CV_16S
                             ptr.method(:get_array_of_int16)
                         when CV_32S
                             ptr.method(:get_array_of_int32)
                         when CV_32F
                             ptr.method(:get_array_of_float32)
                         when CV_64F
                             ptr.method(:get_array_of_float64)
                         else
                             raise "cannot connvert #{self.class} to array"
                         end
                result = []
                0.upto(h-1) do |i|
                    result << getter.call(s*i,w*c)
                end
                result
            end

            def ==(val)
                compare(val,Cv::CMP_EQ)
            end

            def >(val)
                compare(val,Cv::CMP_GT)
            end

            def >=(val)
                compare(val,Cv::CMP_GE)
            end

            def <(val)
                compare(val,Cv::CMP_LT)
            end

            def <=(val)
                compare(val,Cv::CMP_LE)
            end

            def !=(val)
                compare(val,Cv::CMP_NE)
            end

            def ===(val)
                val = compare(val,Cv::CMP_EQ)
                count = cv::countNonZero(val)
                count == val.rows*val.cols
            end

            def compare(val,type)
                val = if val.is_a?(Cv::Mat)
                          val
                      elsif val.respond_to?(:to_a)
                          val.to_a
                      else
                          [val]
                      end
                dst = Cv::Mat.new
                Cv::compare(self,val,dst,type)
                dst
            end
        end
    end
end
