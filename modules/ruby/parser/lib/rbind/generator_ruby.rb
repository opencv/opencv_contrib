require 'fileutils'
require 'delegate'
require 'erb'
require 'open-uri'

module Rbind
    class GeneratorRuby
        include Hooks
        define_hook :on_normalize_default_value

        class << self
            attr_accessor :ruby_default_value_map
            attr_accessor :on_normalize_type_name
            attr_accessor :ffi_type_map
        end
        self.ruby_default_value_map ||= {"true" => "true","TRUE" => "true", "false" => "false","FALSE" => "false"}
        self.ffi_type_map ||= {"char *" => "string","unsigned char" => "uchar" ,"const char *" => "string","uint8_t" => "uint8",
                               "uint32_t" => "uint32","uint64_t" => "uint64","int8_t" => "int8","int32_t" => "int32","int64_t" => "int64" }


        def self.keyword?(name)
            %w{__FILE__ __LINE__ alias and begin BEGIN break case class def defined? do else elsif end END ensure false for if in module next nil not or redo rescue retry return self super then true undef unless until when while yield}.include? name
        end

        def self.normalize_doc(str)
            return nil if !str || str.empty?
            s = str
            str = str.gsub(/\$\$(.*)\$\$/m) do
                "![equation](http://latex.codecogs.com/gif.latex?#{URI::encode($1.to_s)})"
            end
            str = str.gsub(/^ *#?/,"# ")
            "#{str.chomp}\n"
        end

        def self.on_normalize_type_name(&block)
            self.on_normalize_type_name = block
        end

        def self.normalize_arg_name(name)
            name = name.to_s.sub(/\A#{RBase.cprefix}?/, "").gsub(/(?<!\A)\p{Lu}/u, '_\0').downcase
            name = if keyword?(name)
                       "#{name}_"
                   else
                       name
                   end
            # check for digits at the beginning
            if name =~/\d(.*)/
                "_#{name}"
            else
                name
            end
        end

        def self.normalize_default_value(parameter)
            return nil unless parameter.default_value
            if self.callbacks_for_hook(:on_normalize_default_value)
                results = self.run_hook(:on_normalize_default_value,parameter)
                results.compact!
                return results.first unless results.empty?
            end

            val = if parameter.type.basic_type? || parameter.type.ptr?
                      if ruby_default_value_map.has_key?(parameter.default_value)
                          ruby_default_value_map[parameter.default_value]
                      elsif parameter.type.name == "float"
                          parameter.default_value.gsub("f","").gsub(/\.$/,".0").gsub(/^\./,"0.")
                      elsif parameter.type.name == "double"
                          parameter.default_value.gsub(/\.$/,".0").gsub(/^\./,"0.")
                      elsif parameter.type.ptr? &&(parameter.default_value == "0" || parameter.default_value = "NULL")
                          # NULL pointer
                          t = parameter.type.to_raw
                          if t.extern_package_name
                              "::#{t.extern_package_name}::#{normalize_type_name(t.full_name)}::null"
                          else
                              "#{normalize_type_name(t.full_name)}::null"
                          end
                      else
                          normalize_type_name(parameter.default_value)
                      end
                  else
                      if(parameter.default_value.gsub(/^new /,"") =~ /([ \w:<>]*) *\((.*)\)/)
                          value = $2
                          t = parameter.owner.owner.type($1,false)
                          ops = Array(parameter.owner.owner.operation($1,false)) if !t
                          t,ops = if t || !ops.empty?
                                      [t,ops]
                                  else
                                      ns = RBase.namespace($1)
                                      name = RBase.basename($1)
                                      if ns && name
                                          t = parameter.owner.owner.type(ns,false) 
                                          ops = Array(t.operation(name,false)) if t
                                          [t,ops]
                                      else
                                          [nil,nil]
                                      end
                                  end
                          s = if ops && !ops.empty?
                                  if t
                                      if t.extern_package_name
                                          "::#{t.extern_package_name}::#{normalize_type_name(t.full_name)}::#{normalize_method_name(ops.first.name)}(#{(value)})"
                                      else
                                          "#{normalize_type_name(t.full_name)}::#{normalize_method_name(ops.first.name)}(#{(value)})"
                                      end
                                  else
                                      "#{normalize_method_name(ops.first.name)}(#{(value)})"
                                  end
                              elsif t
                                  t = t.to_raw
                                  if t.extern_package_name
                                      "::#{t.extern_package_name}::#{normalize_type_name(t.full_name)}.new(#{(value)})"
                                  else
                                      "#{normalize_type_name(t.full_name)}.new(#{(value)})"
                                  end
                              end
                      else
                          parameter.default_value
                      end
                  end
            if val
                val
            else
               raise "cannot parse default parameter value #{parameter.default_value} for #{parameter.owner.signature}"
            end
        end


        def self.normalize_type_name(name,template=false)
            name.gsub!(" ","")

            # map template classes
            # std::vector<std::string> -> Std::Vector::Std_String
            if name =~ /([\w:]*)<(.*)>$/
                return "#{normalize_type_name($1,true)}::#{normalize_type_name($2,true).gsub("::","_")}"
            else
                name
            end

            # custom normalization
            if @on_normalize_type_name
                n = @on_normalize_type_name.call(name)
                return n if n
            end

            # Parse constant declaration with suffix like 1000000LL
            if name =~ /^([0-9]+)[uUlL]{0,2}/
                name = $1
                return name
            end

            # map all uint ... to Fixnum
            if !template
                if name =~ /^u?int\d*$/ || name =~ /^u?int\d+_t$/
                    return "Fixnum"
                end
            end

            name = name.gsub(/^_/,"")
            names = name.split("::").map do |n|
                n.gsub(/^(\w)(.*)/) do 
                    $1.upcase+$2
                end
            end
            n = names.last.split("_").first
            if n == n.upcase
                return names.join("::")
            end

            name = names.join("::").split("_").map do |n|
                n.gsub(/^(\w)(.*)/) do 
                    $1.upcase+$2
                end
            end.join("_")
        end

        def self.normalize_basic_type_name_ffi(name)
            n = ffi_type_map[name]
            n ||= name
            if n =~ /\*/
                "pointer"
            else
                n
            end
        end

        def self.normalize_alias_method_name(orig_name)
            name = orig_name

            #replace operatorX with the correct ruby operator when 
            #there are overloaded operators
            name = if name =~/^operator(.*)/
                        n = $1
                        if n =~ /\(\)/
                            raise "forbbiden method name #{name}"
                        elsif n =~ /(.*)(\d)/ # check for overloaded operators and use alias name for them
                            alias_name = RNamespace.default_operator_alias[$1]
                            if not alias_name
                                raise ArgumentError, "Normalization failed. Operator: #{$1} unknown"
                            end
                            "#{alias_name}_operator#{$2}"
                        else
                            # this operators does not exist
                            if n == "++" || n == "--"
                                alias_name = RNamespace.default_operator_alias[n]
                                if not alias_name
                                    raise ArgumentError, "Normalization failed. Operator: #{$1} unknown"
                                end
                                "#{alias_name}_operator"
                            else
                                # we can use the c++ name
                                n
                            end
                        end
                   else
                      name
                   end
        end

        # normalize c method to meet ruby conventions
        # see unit tests
        def self.normalize_method_name(orig_name)
            #remove cprefix and replaced _X with #X
            name = orig_name.to_s.gsub(/\A#{RBase.cprefix}/, "") .gsub(/_((?<!\A)\p{Lu})/u, '#\1')
            #replaced X with _x
            name = name.gsub(/(?<!\A)[\p{Lu}\d]/u, '_\0').downcase
            #replaced _x_ with #x#
            name = name.to_s.gsub(/[_#]([a-zA-Z\d])[_#]/u, '#\1#')
            #replaced _x$ with #x
            name = name.to_s.gsub(/[_#]([a-zA-Z\d])$/u, '#\1')
            #replaced ## with _
            name = name.gsub(/##/, '_')
            #replace #xx with _xx
            name = name.gsub(/#([a-zA-Z\d]{2})/, '_\1')
            #remove all remaining #
            name = name.gsub(/#/, '')
            name = normalize_alias_method_name(name)
            raise RuntimeError, "Normalization failed: generated empty name for #{orig_name}" if name.empty?
            name
        end

        def self.normalize_enum_name(name)
            name = GeneratorRuby.normalize_basic_type_name_ffi name
            #to lower and substitute namespace :: with _
            name = name.gsub("::","_")
            name = name.downcase
            name
        end

        class HelperBase
            extend ::Rbind::Logger

            attr_accessor :name
            def initialize(name,root)
                @name = name.to_s
                @root = root
            end

            def full_name
                @root.full_name
            end

            def binding
                Kernel.binding
            end
        end

        class RBindHelper < HelperBase
            attr_reader :compact_namespace

            def initialize(name, root,compact_namespace=true)
                @compact_namespace = compact_namespace
                super(name,root)
            end

            def normalize_t(name)
                t = GeneratorRuby.normalize_type_name name
                if compact_namespace
                    t.gsub(/^#{self.name}::/,"")
                else
                    t
                end
            end

            def add_doc
                GeneratorRuby.normalize_doc(@root.root.doc) if @root.root.doc?
            end

            def normalize_bt(name)
                GeneratorRuby.normalize_basic_type_name_ffi name
            end

            def normalize_enum(name)
                GeneratorRuby.normalize_enum_name(name)
            end

            def normalize_m(name)
                GeneratorRuby.normalize_method_name name
            end

            def library_name
                @root.library_name
            end

            def required_module_names
                @root.required_module_names.map do |name|
                    "require '#{name.downcase}'\n"
                end.join
            end

            def file_prefix
                @root.file_prefix
            end

            def add_accessors
                str = ""
                @root.root.each_type do |t|
                    next if t.basic_type? && !t.is_a?(RNamespace)
                    str += "\n#methods for #{t.full_name}\n"
                    if t.cdelete_method
                        str += "attach_function :#{normalize_m t.cdelete_method},"\
                        ":#{t.cdelete_method},[#{normalize_t(t.full_name)}],:void\n"
                        str += "attach_function :#{normalize_m t.cdelete_method}_struct,"\
                        ":#{t.cdelete_method},[#{normalize_t(t.full_name)}Struct],:void\n"
                    end
                    t.each_operation do |op|
                        return_type = if op.constructor?
                                          "#{normalize_t op.owner.full_name}"
                                      else
                                          op_return_type = op.return_type
                                          if op_return_type.respond_to?(:get_base_delegate)
                                              if klass = op_return_type.get_base_delegate
                                                  op_return_type = klass if klass.kind_of?(REnum)
                                              end
                                          end

                                          if op_return_type.basic_type?
                                              if op_return_type.ptr?
                                                  ":pointer"
                                              elsif op_return_type.kind_of?(REnum)
                                                  ":#{normalize_enum op_return_type.to_raw.csignature}"
                                              else
                                                  ":#{normalize_bt op_return_type.to_raw.csignature}"
                                              end
                                          else
                                              if op_return_type.extern_package_name
                                                  normalize_t("::#{op_return_type.extern_package_name}::#{op_return_type.to_raw.full_name}")
                                              elsif op_return_type.kind_of?(REnum)
                                                  ":#{normalize_enum op_return_type.to_raw.full_name}"
                                              else
                                                  normalize_t op_return_type.to_raw.full_name
                                              end
                                          end
                                      end
                        args = op.cparameters.map do |p|
                            p_type = p.type
                            if p_type.respond_to?(:get_base_delegate)
                                if klass = p_type.get_base_delegate
                                    p_type = klass if klass.kind_of?(REnum)
                                end
                            end
                            if p_type.basic_type?
                                if p_type.ptr? || p.type.ref?
                                    ":pointer"
                                elsif p.type.kind_of?(REnum)
                                    # Includes enums, which need to be defined accordingly
                                    # using ffi:
                                    # enum :normalized_name, [:first, 1,
                                    #                         :second,
                                    #                         :third]
                                    #
                                    ":#{normalize_enum p.type.to_raw.csignature}"
                                else
                                    ":#{normalize_bt p.type.to_raw.csignature}"
                                end
                            else
                                if p_type.extern_package_name
                                    normalize_t("::#{p_type.extern_package_name}::#{p_type.to_raw.full_name}")
                                elsif p_type.kind_of?(REnum)
                                    ":#{normalize_enum p_type.to_raw.full_name}"
                                else
                                    normalize_t p_type.to_raw.full_name
                                end
                            end
                        end
                        fct_name = normalize_m op.cname
                        str += "attach_function :#{fct_name},:#{op.cname},[#{args.join(",")}],#{return_type}\n"
                        str
                    end
                    str+"\n"
                end
                str+"\n"
                str.gsub(/\n/,"\n        ")
            end

            def add_enums
                str = "\n"
                @root.root.each_type do |t|
                    if t.kind_of?(REnum)
                        str += "\tenum :#{GeneratorRuby::normalize_enum_name(t.to_raw.csignature)}, ["
                        t.values.each do |name,value|
                            if value
                                str += ":#{name},#{value}, "
                            else
                                str += ":#{name}, "
                            end
                        end
                        str += "]\n\n"
                    end
                end
                str
            end

        end

        class RTypeTemplateHelper < HelperBase
            def initialize(name, root,compact_namespace = false)
                @type_template_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","ruby","rtype_template.rb")).read,nil,"-")
                super(name,root)
            end

            def name
                GeneratorRuby.normalize_type_name(@name)
            end

            def cname
                GeneratorRuby.normalize_type_name(@root.cname)
            end

            def add_doc
                str = GeneratorRuby::normalize_doc(@root.doc)
            end

            def add_specializing(root = @root)
                root.specialize_ruby
            end

            def result
                return "" if @root.extern?
                str = @type_template_wrapper.result(self.binding)
                str.gsub!("\n","\n    ").gsub!("    \n","\n")
                "    "+str[0,str.size-4]
            end
        end


        class RTypeHelper < HelperBase
            class OperationHelper < SimpleDelegator
                def min_number_of_parameters
                    count = 0
                    parameters.each do |p|
                        break if p.default_value
                        count +=1
                    end
                    count
                end

                def signature_default_values
                    str = parameters.map do |p|
                        if p.default_value
                            GeneratorRuby.normalize_default_value p
                        else
                            "nil"
                        end
                    end.join(", ")
                    "[#{str}]"
                end

                def wrap_parameters_signature
                    parameters.map do |p|
                        n = GeneratorRuby.normalize_arg_name p.full_name
                        if p.default_value 
                            "#{n} = #{GeneratorRuby.normalize_default_value p}"
                        else
                            n
                        end
                    end.join(", ")
                end

                def wrap_parameters_call
                    paras = []
                    paras << "self" if instance_method?
                    paras += parameters.map do |p|
                        GeneratorRuby.normalize_arg_name p.name
                    end
                    paras.join(", ")
                end

                def name
                    if attribute?
                        name = GeneratorRuby.normalize_method_name(attribute.name)
                        if __getobj__.is_a? RGetter
                            name
                        else
                            "#{name}="
                        end
                    else
                        name = if auto_alias
                                   __getobj__.name
                               else
                                   __getobj__.alias || __getobj__.name
                               end
                        GeneratorRuby.normalize_method_name(name)
                    end
                end

                def cname
                    GeneratorRuby.normalize_method_name(__getobj__.cname)
                end

                def add_alias
                    name = if auto_alias
                               __getobj__.name
                           else
                               __getobj__.alias || __getobj__.name
                           end
                    name = GeneratorRuby::normalize_alias_method_name(name)
                    if name == self.name || !cplusplus_alias?
                        nil
                    elsif static?
                        "    class << self; alias :#{name} :#{self.name}; end\n"
                    else
                        "    alias :#{name} :#{self.name}\n"
                    end
                end

                def add_specialize_ruby
                    str = specialize_ruby
                    "    #{str}\n" if str
                end

                def generate_param_doc
                    paras = parameters.map do |p|
                        n = GeneratorRuby.normalize_arg_name p.name
                        t = if p.type.basic_type? && (p.type.ptr? || p.type.ref?)
                                "FFI::MemoryPointer"
                            else
                                GeneratorRuby.normalize_type_name(p.type.full_name)
                            end
                        "# @param [#{t}] #{n} #{p.doc}"
                    end
                    if return_type
                        t = GeneratorRuby.normalize_type_name(return_type.full_name)
                        paras << "# @return [#{t}]"
                    end
                    paras.join("\n")+"\n"
                end

                def add_doc
                    str = GeneratorRuby::normalize_doc(doc)
                    str = if !parameters.empty? || return_type
                              str += "#\n" if str
                              "#{str}#{generate_param_doc}"
                          end
                    str.gsub(/^/,"    ") if str
                end

                def binding
                    Kernel.binding
                end
            end

            class OverloadedOperationHelper < HelperBase
                def initialize(root)
                    raise "expect an array of methods but got #{root}" if root.size < 1
                    super(GeneratorRuby.normalize_method_name(root.first.alias || root.first.name),root)
                    @overload_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","ruby","roverloaded_method_call.rb")).read,nil,"-")
                end

                def static?
                    @root.first.static?
                end

                def add_methods
                    str = @root.map do |method|
                        next if method.ignore?
                        raise "Cannot overload attributes" if method.attribute?
                        op = if method.is_a? OperationHelper
                                 method
                             else
                                 OperationHelper.new(method)
                             end
                        @overload_wrapper.result(op.binding)
                    end.join("\n")
                end

                def add_alias
                    name = GeneratorRuby::normalize_alias_method_name(@root.first.alias || @root.first.name)
                    if name == self.name || !@root.first.cplusplus_alias?
                        nil
                    elsif static?
                        "    class << self; alias :#{name} :#{self.name}; end\n"
                    else
                        "    alias #{name} #{self.name}\n"
                    end
                end

                def add_doc
                    str = @root.map do |op|
                        s = op.add_doc
                        s ||= ""
                        s = s.gsub(/( *#)/) do
                            "#{$1}  "
                        end
                        "    # @overload #{op.name}(#{op.wrap_parameters_signature})\n#{s}"
                    end.join("    #\n")
                end

                def binding
                    Kernel.binding
                end
            end

            def initialize(name, root,compact_namespace = false)
                @type_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","ruby","rtype.rb")).read,nil,"-")
                @namespace_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","ruby","rnamespace.rb")).read,nil,"-")
                @static_method_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","ruby","rstatic_method.rb")).read,nil,"-")
                @method_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","ruby","rmethod.rb")).read,nil,'-')
                @overloaded_method_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","ruby","roverloaded_method.rb")).read,nil,"-")
                @overloaded_static_method_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","ruby","roverloaded_static_method.rb")).read,nil,"-")
                @overloaded_method_call_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","ruby","roverloaded_method_call.rb")).read,nil,"-")
                @compact_namespace = compact_namespace
                super(name,root)
            end

            def name
                GeneratorRuby.normalize_type_name(@name)
            end

            def cname
                GeneratorRuby.normalize_type_name(@root.cname)
            end

            def cdelete_method
                GeneratorRuby.normalize_method_name(@root.cdelete_method)
            end

            def add_specializing(root = @root)
                str = root.specialize_ruby.to_s
                root.each_type(false) do |t|
                    next if t.basic_type? && !t.is_a?(RNamespace)
                    str += add_specialize(t) if name == GeneratorRuby.normalize_type_name(t.full_name)
                end
                str
            end

            def add_constructor
                raise "there is no constructor for namespaces!" if self.is_a?(RNamespace)
                ops = Array(@root.operation(@root.name,false))
                ops.map do |c|
                    next if c.ignore?
                    ch = OperationHelper.new(c)
                    @overloaded_method_call_wrapper.result(ch.binding)
                end.join("\n")
            end

            def add_constructor_doc
                ops = Array(@root.operation(@root.name,false))
                ops = ops.map do |c|
                    next if c.ignore?
                    OperationHelper.new(c)
                end.compact
                if ops.empty?
                    nil
                elsif ops.size == 1
                    ops.first.add_doc
                else
                    OverloadedOperationHelper.new(ops).add_doc
                end
            end

            def add_consts(root=@root)
                str = @root.consts.map do |c|
                    next if c.extern? || c.ignore?
                    if not c.default_value
                        HelperBase.log.warn "#{c.name}: no default value"
                        next
                    else
                    "    #{c.name} = #{GeneratorRuby::normalize_type_name(c.default_value)}\n"
                    end
                end.join
                return str unless @compact_namespace

                root.each_type(false) do |t|
                    next if t.basic_type? && !t.is_a?(RNamespace)
                    str += add_consts(t) if name == GeneratorRuby.normalize_type_name(t.full_name)
                end
                str
            end

            def add_to_s
                str = []
                @root.each_operation do |o|
                    next unless o.is_a? RGetter
                    op = OperationHelper.new(o)
                    str << "#{op.name}=\#{self.#{op.name}}"
                end
                "\"#<#{full_name} #{str.join(" ")}>\""
            end

            def add_methods(root=@root)
                # sort all method according their target name
                ops = Hash.new do |h,k|
                    h[k] = Array.new
                end
                root.each_operation do |o|
                    begin
                        next if o.constructor? || o.ignore?
                        op = OperationHelper.new(o)
                        if op.instance_method?
                            ops["rbind_instance_#{op.name}"] << op
                        else
                            ops["rbind_static_#{op.name}"] << op
                        end
                    rescue => e
                        HelperBase.log.warn "Operation '#{o}' not added. #{e}"
                    end
                end
                # render method
                str = ""
                ops.each_value do |o|
                    begin
                        if o.size == 1
                            op = o.first
                            str += if op.instance_method?
                                       @method_wrapper.result(op.binding)
                                   else
                                       @static_method_wrapper.result(op.binding)
                                   end
                        else
                            helper = OverloadedOperationHelper.new(o)
                            str += if o.first.instance_method?
                                       @overloaded_method_wrapper.result(helper.binding)
                                   else
                                       @overloaded_static_method_wrapper.result(helper.binding)
                                   end
                        end
                    rescue => e
                        pp e
                        HelperBase.log.error "Operation '#{o}' could not be rendered. #{e}"
                    end
                end
                return str unless @compact_namespace
                root.each_type(false) do |t|
                    next if t.basic_type? && !t.is_a?(RNamespace)
                    str += add_methods(t) if name == GeneratorRuby.normalize_type_name(t.full_name)
                end
                str
            end

            def add_types(root = @root)
                str = ""
                root.each_type(false,true) do |t|
                    next if t.ignore? || t.extern?
                    next if t.basic_type? && !t.is_a?(RNamespace)
                    str += if @compact_namespace && name == GeneratorRuby.normalize_type_name(t.full_name)
                               add_types(t)
                           elsif t.template?
                               RTypeTemplateHelper.new(t.name,t).result
                           else
                               RTypeHelper.new(t.name,t).result
                           end
                end
                str
            end

            def full_name
                @root.full_name
            end

            def add_doc
                GeneratorRuby::normalize_doc(@root.doc)
            end

            def result
                return "" if @root.extern?
                str = if @root.is_a? RClass
                          @type_wrapper.result(self.binding)
                      else
                          @namespace_wrapper.result(self.binding)
                      end
                if(@root.root?)
                    str
                else
                    str.gsub!("\n","\n    ").gsub!("    \n","\n")
                    "    "+str[0,str.size-4]
                end
            end
        end

        attr_accessor :module_name
        attr_accessor :required_module_names
        attr_accessor :library_name
        attr_accessor :output_path
        attr_accessor :file_prefix
        attr_accessor :compact_namespace
        attr_reader :root

        def file_prefix
            @file_prefix || GeneratorRuby.normalize_method_name(module_name)
        end

        def initialize(root,module_name ="Rbind",library_name="rbind_lib")
            @root = root
            @rbind_wrapper = ERB.new(File.open(File.join(File.dirname(__FILE__),"templates","ruby","rbind.rb")).read,nil,"-")
            @module_name = module_name
            @library_name = library_name
            @compact_namespace = true
        end

        def generate(path=@output_path)
            @output_path = path
            FileUtils.mkdir_p(path) if path  && !File.directory?(path)
            file_rbind = File.new(File.join(path,"#{file_prefix}.rb"),"w")
            file_types = File.new(File.join(path,"#{file_prefix}_types.rb"),"w")

            types = RTypeHelper.new(@module_name,@root,compact_namespace)
            file_types.write types.result
            rbind = RBindHelper.new(@module_name,self,compact_namespace)
            file_rbind.write @rbind_wrapper.result(rbind.binding)
        end
    end
end
