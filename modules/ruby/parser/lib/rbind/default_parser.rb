
module Rbind
    class DefaultParser < RNamespace
        extend ::Rbind::Logger

        def initialize(root = nil)
            super(nil,root)
            add_default_types if !root
        end

        def normalize_flags(line_number,flags,*valid_flags)
            flags.map do |flag|
                next if flag.empty?
                if flag =~ /(\w*)(.*)/
                    DefaultParser.log.debug "input line #{line_number}: ignoring flag #{$2}" unless $2.empty?
                    flag = $1.to_sym
                    DefaultParser.log.debug "input line #{line_number}: ignoring flag #{$1}" unless valid_flags.include?(flag)
                    flag
                else
                    raise "cannot parse flag #{flag.inspect}"
                end
            end.compact
        end

        def add_data_type_name(name)
            t = RDataType.new(name)
            add_type t
            t
        end

        def add_namespace_name(name)
            ns = RNamespace.new(name)
            add_type ns
            ns
        end

        def add_class_name(name)
            klass = RClass.new(name)
            add_type klass
            klass
        end

        def on_type_not_found(&block)
            @on_type_not_found = block
        end

        # reverse template masking done by the opencv parser
        def unmask_template(type_name)
            if(type_name =~/<.*>/)
                return type_name
            end

            if(type_name =~/^vector/ || type_name =~/^Ptr/)
               if(type_name =~ /^([a-zA-Z\d]*)_([_a-zA-Z\d]*) ?(\(?.*)\)? */)
                  "#{$1}<#{unmask_template($2)}>#{$3}"
               else
                   type_name
               end
            else
                type_name
            end
        end

        def find_type(owner,type_name,braise=true)
            type_name = unmask_template(type_name)
            t = owner.type(type_name,false)
            return t if t

            normalized = type_name.split("_")
            name = normalized.shift
            while !normalized.empty?
                name += "::#{normalized.shift}"
                t = if normalized.empty?
                        owner.type(name,false)
                    else
                        owner.type("#{name}_#{normalized.join("_")}",false)
                    end
                return t if t
            end
            t = @on_type_not_found.call(owner,type_name) if @on_type_not_found
            return t if t

            #search again even if we know the type is not there to create a proper error message
            owner.type(type_name,true) if braise
        end

        def parameter(line_number,string,owner = self)
            flags = string.split(" /")
            array = flags.shift.split(" ")
            type_name = array.shift
            para_name = array.shift
            default = unmask_template(array.join(" "))
            type = find_type(owner,type_name)
            flags = normalize_flags(line_number,flags,:IO,:O)
            type = if flags.include?(:O) || flags.include?(:IO)
                       if type.ptr?
                           type
                       else
                           type.to_ref  # the opencv parser encodes references as flags
                       end
                   elsif type.basic_type?
                       type
                   else
                       type.to_const
                   end
            RParameter.new(para_name,type,default)
        rescue RuntimeError => e
            raise "input line #{line_number}: #{e}"
        end

        def attribute(line_number,string,owner=self)
            flags = string.split(" /")
            array = flags.shift.split(" ")
            type_name = array[0]
            name = array[1]
            type = find_type(owner,type_name,false)
            # auto embedded types
            type ||= begin
                         t = owner.add_type(RClass.new(RBase.normalize(type_name)))
                         t.extern_package_name = @extern_package_name
                         t
                     end
            flags = normalize_flags(line_number,flags,:RW,:R)
            a = RAttribute.new(name,type)
            a.writeable!(true) if flags.include? :RW
            a
        rescue RuntimeError => e
            raise "input line #{line_number}: #{e}"
        end

        def parse_class(line_number,string)
            lines = string.split("\n")
            a = lines.shift.rstrip
            unless a =~ /class ([<>a-zA-Z\.\d_:]*) ?:?([<>a-zA-Z\.\:, \d_]*)(.*)/
                raise "cannot parse class #{a}"
            end
            name = $1
            parent_classes = $2
            flags = $3
            parent_classes = if parent_classes
                                 parent_classes.gsub(" ","").split(",").map do |name|
                                     #TODO this should also call the user callback
                                     t = type(RBase.normalize(name),false)
                                     # remove namespace and try again
                                     # this is workaround for the hdr_parser adding 
                                     # always the namespace to the parent class
                                     t ||= begin
                                               names = name
                                               while((names=RBase.normalize(names).split("::")).size > 1)
                                                     names.shift
                                                     names = names.join("::")
                                                     t = type(names,false)
                                                     break t if t
                                               end
                                           end
                                     # auto add parent class
                                     t ||= begin
                                               t = add_type(RClass.new(RBase.normalize(name)))
                                               t.extern_package_name = @extern_package_name
                                               t
                                           end
                                 end
                             end
            flags = if flags
                       normalize_flags(line_number,flags.gsub(" ","").split("/").compact,:Simple,:Map)
                    end
            t = RClass.new(name,*parent_classes)
            t.extern_package_name = @extern_package_name
            t = add_type(t)
            line_counter = 1
            lines.each do |line|
                a = attribute(line_counter+line_number,line,t)
                t.add_attribute(a)
                line_counter += 1
            end
            [t,line_counter]
        rescue RuntimeError  => e
            raise "input line #{line_number}: #{e}"
        end

        def add_type(klass)
            if klass2 = type(klass.full_name,false)
                if !klass2.is_a?(RClass) || (!klass2.parent_classes.empty? && klass2.parent_classes != klass.parent_classes)
                    raise "Cannot add class #{klass.full_name}. A different type #{klass2} is already registered"
                else
                    klass.parent_classes.each do |p|
                        klass2.add_parent p
                    end
                    klass2.extern_package_name = klass.extern_package_name
                    klass2
                end
            else
                klass.name = klass.name.gsub(">>","> >")
                super(klass)
                klass
            end
        end

        def parse_struct(line_number,string)
            a = string.split("\n")
            first_line = a.shift
            flags = first_line.split(" /")
            name = flags.shift.split(" ")[1]
            flags = normalize_flags(line_number,flags)
            klass = RClass.new(name)
            klass = add_type(klass)
            line_counter = 1
            a.each do |line|
                a = attribute(line_counter+line_number,line,klass)
                klass.add_attribute(a)
                line_counter += 1
            end
            klass.extern_package_name = @extern_package_name
            [klass,line_counter]
        rescue RuntimeError  => e
            raise "input line #{line_number}: #{e}"
        end

        def parse_const(line_number,string)
            raise "multi line const are not supported: #{string}" if string.split("\n").size > 1
            unless string =~ /const ([a-zA-Z\.\d_:]*) ?([^\/]*)(.*)/
                raise "cannot parse const #{string}"
            end
            name = $1
            value = $2.chomp("\n").chomp(" ")
            flags = $3
            flags = if flags
                       normalize_flags(line_number,flags.gsub(" ","").split("/").compact)
                    end

            c = RParameter.new(name,find_type(self,"const int"),value)
            c.extern_package_name = @extern_package_name
            add_const(c)
            [c,1]
        end

        def parse_operation(line_number,string)
            a = string.split("\n")
            line = a.shift
            flags = line.split(" /")
            line = flags.shift
            elements = line.split(" ")
            name = elements.shift
            return_type_name = elements.shift
            if return_type_name == "()"
                name += return_type_name
                return_type_name = elements.shift
            end
            alias_name = elements.shift
            alias_name = if alias_name
                             raise "#{line_number}: cannot parse #{string}" unless alias_name =~/^=.*/
                             alias_name.gsub("=","")
                         end

            ns = RBase.namespace(name)
            owner = type(ns,false)
            owner ||= add_namespace_name(ns)
            if return_type_name == "explicit"
                flags << return_type_name
                return_type_name = nil
            end
            flags = normalize_flags(line_number,flags,:S,:O)
            return_type = if return_type_name && !return_type_name.empty?
                              t = find_type(owner,return_type_name)
                              if !t.ptr? && flags.include?(:O)
                                  t.to_ref
                              else
                                  t
                              end
                          end
            line_counter = 1
            args = a.map do |line|
                p = parameter(line_number+line_counter,line,owner)
                line_counter += 1
                p
            end
            op = ::Rbind::ROperation.new(name,return_type,*args)
            op.alias = alias_name if alias_name && !alias_name.empty?
            op = if flags.include?(:S)
                     op.to_static
                 else
                     op
                 end
            type(op.namespace,true).add_operation(op)
            op.extern_package_name = @extern_package_name
            [op,line_counter]
        end

        def parse(string,extern_package_name=nil)
            @extern_package_name = extern_package_name

            a = split(string)
            #remove number at the end of the file
            a.pop if a.last.to_i != 0

            line_number = 1
            a.each do |block|
                begin
                first = block.split(" ",2)[0]
                obj,lines = if first == "const"
                                parse_const(line_number,block)
                            elsif first == "class"
                                parse_class(line_number,block)
                            elsif first == "struct"
                                parse_struct(line_number,block)
                            else
                                parse_operation(line_number,block)
                            end
                line_number+=lines
                rescue RuntimeError => e
                    puts "Parsing Error: #{e}"
                    puts "Line #{line_number}:"
                    puts "--------------------------------------------------"
                    puts block
                    puts "--------------------------------------------------"
                    Kernel.raise
                    break
                end
            end
            a.size
        end

        def split(string)
            array = []
            if string
                string.each_line do |line|
                    if line == "\n" || line.empty?
                       next
                    elsif line[0] != " "
                        array << line
                    elsif !array.empty?
                        array[array.size-1] = array.last + line
                    end
                end
            end
            array
        end
    end
end
