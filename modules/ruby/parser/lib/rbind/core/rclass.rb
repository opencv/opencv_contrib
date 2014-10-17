
module Rbind
    class RClass < RNamespace
        attr_reader :parent_classes
        attr_reader :attributes
        ParentClass = Struct.new(:type,:accessor)

        def initialize(name,*parent_classes)
            @parent_classes  = Hash.new
            @attributes = Hash.new
            parent_classes.flatten!
            parent_classes.each do |p|
                add_parent(p)
            end
            super(name)
            # we have to disable the type check for classes
            # otherwise derived types cannot be parsed
            @check_type = false
        end

        def basic_type?
            false
        end

        def constructor?
            ops = Array(operation(name,false))
            return false unless ops
            op = ops.find do |op|
                op.constructor?
            end
            !!op
        end

        def add_attribute(attr)
            if attr.namespace?
                type(attr.namespace).add_attribute(attr)
            else
                if @attributes.has_key? attr.name
                    raise "#An attribute with the name #{attr.name} already exists"
                end
                attr.owner = self
                @attributes[attr.name] = attr
                # add getter and setter methods to the object
                add_operation(RGetter.new(attr)) if attr.readable?
                add_operation(RSetter.new(attr)) if attr.writeable?
            end
            self
        end

        def attributes
            attribs = @attributes.values
            parent_classes.each do |k|
                others = k.type.attributes
                others.delete_if do |other|
                    attribs.inclue? other
                end
                others = others.map(&:dup)
                others.each do |other|
                    other.owner = self
                end
                attribs += others
            end
            attribs
        end

        def attribute(name)
            attrib = @attributes[name]
            attrib ||= begin
                           p = parent_classes.find do |k|
                               k.type.attribute(name)
                           end
                           a = p.attribute(name).dup if p
                           a.owner = self if a
                           a
                       end
        end

        def operations
            # temporarily add all base class operations
            own_ops = @operations.dup
            parent_classes.each do |k|
                k.type.operations.each do |other_ops|
                    next if other_ops.empty?
                    ops = if @operations.has_key?(other_ops.first.name)
                            @operations[other_ops.first.name]
                          else
                              []
                          end
                    other_ops.delete_if do |other_op|
                        next true if !other_op || other_op.constructor?
                        op = ops.find do |o|
                            o == other_op
                        end
                        next false if !op
                        next true if op.base_class == self
                        next true if op.base_class == other_op.base_class
                        # ambiguous name look up due to multi
                        # inheritance
                        op.ambiguous_name = true
                        other_op.ambiguous_name = true
                        false
                    end
                    other_ops = other_ops.map(&:dup)
                    other_ops.each do |other|
                        old = other.alias
                        add_operation other
                    end
                end
            end
            # copy embedded arrays other wise they might get modified outside
            result = @operations.values.map(&:dup)
            @operations = own_ops
            result
        end

        def used_namespaces
            namespaces = super.clone
            parent_classes.each do |k|
                namespaces += k.type.used_namespaces
            end
            namespaces
        end

        def operation(name,raise_=true)
            ops = if @operations.has_key? name
                      @operations[name].dup
                  else
                      []
                  end
            parent_classes.each do |k|
                other_ops = Array(k.type.operation(name,false))
                other_ops.delete_if do |other_op|
                    ops.include? other_op
                end
                ops += other_ops
            end
            if(ops.size == 1)
                ops.first
            elsif ops.empty?
                raise "#{full_name} has no operation called #{name}." if raise_
            else
                ops
            end
        end

        def cdelete_method
            if @cdelete_method
                @cdelete_method
            else
                if cname =~ /^#{RBase.cprefix}(.*)/
                    "#{RBase.cprefix}delete_#{$1}"
                else
                    "#{RBase.cprefix}delete_#{name}"
                end
            end
        end

        def empty?
            super && parent_classes.empty? && attributes.empty?
        end

        def pretty_print_name
            str = "#{"template " if template?}class #{full_name}"
            unless parent_classes.empty?
                parents = parent_classes.map do |p|
                    p.type.full_name
                end
                str += " : " +  parents.join(", ")
            end
            str
        end

        def pretty_print(pp)
            super
            unless attributes.empty?
                pp.nest(2) do
                    pp.breakable
                    pp.text "Attributes:"
                    pp.nest(2) do
                        attributes.each do |a|
                            pp.breakable
                            pp.pp(a)
                        end
                    end
                end
            end
        end

        def add_parent(klass,accessor=:public)
            klass,accessor = if klass.is_a?(ParentClass)
                                 [klass.type,klass.accessor]
                             else
                                 [klass,accessor]
                             end
            if @parent_classes.has_key? klass.name
                raise ArgumentError,"#A parent class with the name #{klass.name} already exists"
            end
            if klass.full_name == full_name || klass == self
                raise ArgumentError,"class #{klass.full_name} cannot be parent of its self"
            end
            @parent_classes[klass.name] = ParentClass.new(klass,accessor)
            self
        end

        def parent_class(name)
            @parent_class[name].type
        end

        def parent_classes(accessor = :public)
            @parent_classes.values.find_all do |k|
                k.accessor = accessor
            end
        end
    end
end
