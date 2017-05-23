module Rbind
    class RNamespace< RDataType
        include Hooks
        define_hook :on_type_not_found
        define_hook :on_type_look_up
        define_hook :on_add_type
        define_hook :on_add_operation
        define_hook :on_add_const

        class << self
            attr_accessor :default_type_names
            attr_accessor :default_type_alias
            attr_accessor :default_operator_alias
        end
        # TODO move somewhere else
        self.default_type_names = [:int,:int8,:int32,:int64,:uint,:uint8,:uint32,:uint64,
                                   :int8_t,:int32_t,:int64_t,:uint8_t,:uint32_t,:uint64_t,
                                   :bool,:double,:float,:void,:char,:size_t,:long,
                                   :uchar, :char16, :char32, :ushort, :ulong, :ulong_long,
                                   :uint128, :short, :long_long, :int128, :long_double,
                                   :c_string,:constant_array]
        self.default_type_alias= { :u_char => :uchar, :u_short => :ushort, :u_long => :ulong,
                                   :u_long_long => :ulong_long,:u_int => :uint, :uint128 => :uint128,
                                   :char_s => :char}

        self.default_operator_alias = {
                "()" => "fct",
                "!=" => "unequal",
                "==" => "equal",
                "&=" => "and_set",
                "+=" => "add",
                "-=" => "sub",
                "++"  => "plusplus",
                "--"  => "minusminus",
                "[]" => "array",
                "<=" => "less_or_equal",
                "<"  => "less",
                ">=" => "greater_or_equal",
                "+"  => "plus",
                "-"  => "minus",
                "*"  => "mult",
                "/"  => "div",
                "!"  => "not",
                "&"  => "and",
                ">"  => "greater",
                "="  => "assign"
            }


        attr_reader :operations
        attr_reader :operation_alias
        attr_reader :consts
        attr_reader :used_namespaces
        attr_reader :type_alias
        attr_reader :root

        def initialize(name=nil,root = nil)
            @consts = Hash.new
            @types = Hash.new
            @type_alias = Hash.new
            @operations = Hash.new{|hash,key| hash[key] = Array.new}
            @operation_alias = Hash.new{|hash,key| hash[key] = Array.new}
            @used_namespaces = Array.new
            name ||= begin
                         @root = true
                         "root"
                     end
            if root
                #share varaibales accross root namesapces
                raise "#{root} is not a root namespace" unless root.root?
                @consts = root.instance_variable_get(:@consts)
                @types = root.instance_variable_get(:@types)
                @type_alias = root.instance_variable_get(:@type_alias)
                @operations = root.instance_variable_get(:@operations)
                @operation_alias =root.instance_variable_get(:@operation_alias)
                @used_namespaces = root.instance_variable_get(:@used_namespaces)
            end
            super(name)
        end

        def root?
            !!root
        end

        def constructor?
            false
        end

        def types
            @types.values
        end

        def delete_type(name)
            ns = RBase.namespace(name)
            if ns
                type(ns).delete_type(RBase.basename(name))
            else
                @types.delete(name)
            end
        end

        def use_namespace(namespace)
            # Check if there is another root embedded
            # and add the corresponding namespace.
            # This is needed to support multiple roots providing
            # types to the same namespace
            @used_namespaces.each do |ns|
                next unless ns.root?
                if (ns2 = ns.type(namespace.full_name,false))
                    @used_namespaces << ns2 if ns.object_id != namespace.object_id
                end
            end
            @used_namespaces << namespace
        end

        def each_type(childs=true,all=false,&block)
            if block_given?
                types.each do |t|
                    next if !all && (t.ignore? || t.extern? || t.template?)
                    yield t
                    t.each_type(childs,all,&block) if childs && t.respond_to?(:each_type)
                end
            else
                Enumerator.new(self,:each_type,childs,all)
            end
        end

        def each_container(childs=true,all=false,&block)
            each_type(childs,all) do |t|
                next unless t.container?
                yield t
            end
        end

        def consts
            @consts.values
        end

        def const(name,raise_ = true,search_owner = true)
            c = if @consts.has_key?(name)
                    @consts[name]
                else
                    if !!(ns = RBase.namespace(name))
                        t = type(ns,false)
                        t.const(RBase.basename(name),false,false) if t
                    end
                end
            c ||= begin
                      used_namespaces.each do |ns|
                          c = ns.const(name,false,false)
                          break if c
                      end
                      c
                  end
            c ||= if search_owner && owner
                      owner.const(name,false)
                  end
            raise RuntimeError,"#{full_name} has no const called #{name}" if raise_ && !c
            c
        end

        def each_const(childs=true,all=false,&block)
            if block_given?
                consts.each do |c|
                    next if !all && (c.ignore? || c.extern?)
                    yield c
                end
                return unless childs
                each_container(false,all) do |t|
                    t.each_const(childs,all,&block)
                end
            else
                Enumerator.new(self,:each_const,childs,all)
            end
        end

        def extern?
            return super if self.is_a? RClass

            # check if self is container holding only
            # extern objects
            each_type(false) do |t|
                return false if !t.extern?
            end
            each_const(false) do |c|
                return false if !c.extern?
            end
            each_operation do |t|
                return false
            end
        end

        def each_operation(all=false,&block)
            if block_given?
                operations.each do |ops|
                    ops.each do |o|
                        next if !all && o.ignore?
                        yield o
                    end
                end
            else
                Enumerator.new(self,:each_operaion,all)
            end
        end

        def used_extern_types
            extern = Set.new
            each_operation do |op|
                extern << op.return_type.to_raw if op.return_type && op.return_type.extern?
                op.parameters.each do |arg|
                    extern << arg.type.to_raw if arg.type.extern?
                end
            end
            each_container do |type|
                extern += type.used_extern_types
            end
            extern
        end

        def operations
            @operations.values
        end

        def container?
            true
        end

        def operation(name,raise_=true)
            ops = @operations[name]
            if(ops.size == 1)
                ops.first
            elsif ops.empty?
                @operations.delete name
                raise "#{full_name} has no operation called #{name}." if raise_
            else
                ops
            end
        end

        def operation?(name)
            !!operation(name,false)
        end

        def add_operation(op,&block)
            if op.is_a? String
                op = ROperation.new(op,void)
                op.owner = self
                instance_exec(op,&block) if block
                return add_operation(op)
            end

            op.owner = self
            # make sure there is no name clash
            other = @operations[op.name].find do |o|
                o.cname == op.cname
            end
            other ||= @operation_alias[op.name].find do |o|
                o.cname == op.cname
            end
            op.alias = if !other
                           op.alias
                       elsif op.alias
                           name = "#{op.alias}#{@operations[op.name].size+1}"
                           ::Rbind.log.debug "add_operation: name clash: aliasing #{op.alias} --> #{name}"
                           name
                       else
                           op.auto_alias = true
                           name = "#{op.name}#{@operations[op.name].size+1}"
                           ::Rbind.log.debug "add_operation: name clash: #{op.name} --> #{name}"
                           name
                       end
            op.index = @operations[op.name].size
            @operations[op.name] << op
            @operation_alias[op.alias] << op if op.alias
            op
        end

        # TODO rename to add_namespace_name
        def add_namespace(namespace_name)
            names = namespace_name.split("::")
            current_type = self
            while !names.empty?
                name = names.shift
                temp = current_type.type(name,false,false)
                current_type = if temp
                                   temp
                               else
                                   ::Rbind.log.debug "missing namespace: add #{current_type.full_name unless current_type.root?}::#{name}"
                                   current_type.add_type(RNamespace.new(name))
                               end
            end
            current_type
        end

        def add_const(const)
            const.const!
            if(c = const(const.full_name,false,false))
                if c.type.full_name == const.type.full_name && c.default_value == const.default_value
                    ::Rbind.log.warn "A const with the name #{const.full_name} already exists"
                    return c
                else
                    raise ArgumentError,"#A different const with the name #{const.full_name} already exists: #{const} != #{c}"
                end
            end
            if const.namespace? && self.full_name != const.namespace
                t=type(const.namespace,false)
                t ||= add_namespace(const.namespace)
                t.add_const(const)
            else
                const.owner = self
                @consts[const.name] = const
            end
            const
        end

        def add_default_types
            add_simple_types RNamespace.default_type_names
            type("uchar").cname("unsigned char")
            type("c_string").cname("char *")
            RNamespace.default_type_alias.each_pair do |key,val|
                next if @type_alias.has_key?(key)
                @type_alias[key.to_s] = type(val)
            end
            self
        end

        def add_simple_type(name)
            add_type(RDataType.new(name))
        end

        def add_simple_types(*names)
            names.flatten!
            names.each do |n|
                add_simple_type(n)
            end
        end

        def add_type(type,check_exist = true)
            if check_exist && type(type.full_name,false,false)
                raise ArgumentError,"A type with the name #{type.full_name} already exists"
            end
            # if self is not the right namespace
            if type.namespace? && self.full_name != type.namespace && !(self.full_name =~/(.*)::#{type.namespace}/)
                t=type(type.namespace,false)
                t ||=add_namespace(type.namespace)
                t.add_type(type)
            else
                type.owner = self
                if type.alias
                    add_type_alias(type, type.alias, check_exist)
                end
                raise ArgumentError,"A type with the name #{t.full_name} already exists" if(t = @types[type.name])
                @types[type.name] = type.to_raw
            end
            type
        end

        def type?(name,search_owner = true)
            !type(name,false,search_owner).nil?
        end

        def type(name,raise_ = true,search_owner = true)
            name = name.to_s
            constant = if name =~ /const /
                           true
                       else
                           false
                       end
            name = name.gsub("unsigned ","u").gsub("const ","").gsub(" ","").gsub(">>","> >")

            t = if @types.has_key?(name)
                    @types[name]
                elsif @type_alias.has_key?(name)
                    @type_alias[name]
                else
                    if !!(ns = RBase.namespace(name))
                        ns = ns.split("::")
                        ns << RBase.basename(name)
                        t = type(ns.shift,false,false)
                        t.type(ns.join("::"),false,false) if t
                    end
                end
            t ||= begin
                      used_namespaces.each do |ns|
                          t = ns.type(name,false,false)
                          break if t
                      end
                      t
                  end
            t ||= if search_owner && owner
                      owner.type(name,false)
                  end
            # check if type is a pointer and pointee is registered
            t ||= begin
                      ptr_level = $1.to_s.size if name  =~ /(\**)$/
                      name2 = name.gsub("*","")
                      ref_level = $1.to_s.size if name2  =~ /(&*)$/
                      name2 = name2.gsub("&","")
                      if ptr_level > 0 || ref_level > 0
                          t = type(name2,raise_,search_owner)
                          if t
                              1.upto(ptr_level) do
                                  t = t.to_ptr
                              end
                              1.upto(ref_level) do
                                  t = t.to_ref
                              end
                             # t.owner.add_type(t,false)
                              t
                          end
                      end
                  end

            # check if type is a template and a template is registered
            # under the given name
            t ||=  if search_owner && name =~ /([:\w]*)<(.*)>$/
                      t = type($1,false) if $1 && !$1.empty?
                      t2 = type($2,false) if $2 && !$2.empty?
                      # template is known to this library
                      if t && t2 
                          name = "#{t.name}<#{t2.full_name}>"
                          t3 ||= t.owner.type(name,false,false)
                          t3 ||= begin
                                     if !t.template?
                                         ::Rbind.log.error "try to spezialize a class #{t.full_name} --> #{name}"
                                         nil
                                     else
                                         t3 = t.do_specialize(name,t2)
                                         if !t3
                                             ::Rbind.log.error "Failed to specialize template #{t.full_name} --> #{name}"
                                             nil
                                         else
                                             t.owner.add_type(t3,false)
                                             ::Rbind.log.info "spezialize template #{t.full_name} --> #{t3.full_name}"
                                             t3
                                         end
                                     end
                                 end
                      # generic template is unknown
                      # check if specialised template is known
                      elsif t2 && $1 && !$1.empty? 
                          real_name = "#{$1}<#{t2.full_name}>".gsub(">>","> >")
                          type(real_name,false) if(real_name != name) # prevent stack level too deep
                      else
                          nil
                      end
                    end
            if !t && raise_
                if RNamespace.callbacks_for_hook(:on_type_not_found)
                    results = RNamespace.run_hook(:on_type_not_found,self,name)
                    t = results.find do |t|
                        t.respond_to?(:type)
                    end
                end
                raise RuntimeError,"#{full_name} has no type called #{name}" if !t
            end
            if constant
                t.to_const
            else
                t
            end
        end

        def pretty_print_name
            "namespace #{full_name}"
        end

        def root?
            !!root
        end

        def empty?
            consts.empty? && types.empty? && operations.empty?
        end

        def pretty_print(pp)
            pp.text pretty_print_name unless root?

            unless consts.empty?
                pp.nest(2) do
                    pp.breakable
                    pp.text "Consts:"
                    pp.nest(2) do
                        consts.each do |c|
                            pp.breakable
                            pp.pp(c)
                        end
                    end
                end
            end
            unless types.empty?
                pp.nest(2) do
                    pp.breakable
                    pp.text "Types:"
                    pp.nest(2) do
                        simple = []
                        other = []
                        types.each do |t|
                            if t.basic_type? && !t.container?
                                simple << t.name
                            else 
                                other << t
                            end
                        end
                        if !simple.empty?
                            pp.breakable
                            pp.text(simple.join(", "))
                        end
                        other.each do |t|
                            pp.breakable
                            pp.pp(t)
                        end
                    end
                end
            end

            unless operations.empty?
                pp.nest(2) do
                    pp.breakable
                    pp.text "Operations:"
                    pp.nest(2) do
                        operations.each do |op|
                            op.each do |o|
                                pp.breakable
                                pp.pp(o)
                            end
                        end
                    end
                end
            end
        end

        # Adding a type alias in the main namespace, e.g.
        # to register existing typedefs
        #
        # raises if the alias is already register under a given typename
        def add_type_alias(known_type, alias_name, check_exist = true)
            if check_exist && type_alias.has_key?(alias_name)
                existing_type = type(alias_name, false, false)
                if known_type.name == existing_type.name
                    ::Rbind.log.warn "Alias: '#{alias_name} for type '#{known_type.name}' already registered'"
                else
                    raise ArgumentError, "Cannot register alias: #{alias_name} for type '#{known_type.name}'. Alias already registered with type: '#{existing_type.name}'"
                end
            else
                ::Rbind.log.debug "Alias: '#{alias_name} for type '#{known_type.name}' registered"
                @type_alias[alias_name] = known_type.to_raw
            end
        end

        def self.default_operators
            default_operator_alias.keys
        end

        def method_missing(m,*args)
            if m != :to_ary && args.empty?
                t = type(m.to_s,false,false)
                return t if t

                op = operation(m.to_s,false)
                return op if op

                c = const(m.to_s,false)
                return c if c
            end
            super
        end
    end
end



