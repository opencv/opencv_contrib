module Rbind
    class StdVector < RTemplateClass
        def specialize(klass,*parameters)
            if parameters.size != 1
                raise ArgumentError,"StdVector does only support one template parameter. Got: #{parameters}}"
            end
            vector_type = parameters.flatten.first

            klass.add_operation ROperation.new(klass.name,nil)
            klass.add_operation ROperation.new(klass.name,nil,RParameter.new("other",klass).to_const)

            para = Array.new
            para <<  RParameter.new("size",type("size_t"))
            para <<  RParameter.new("val",vector_type).default_value(vector_type.full_name+"()").to_const
            klass.add_operation ROperation.new("resize",type("void"),para)
            klass.add_operation ROperation.new("size",type("size_t"))
            klass.add_operation ROperation.new("clear",type("void"))
            klass.add_operation ROperation.new("capacity",type("size_t"))
            klass.add_operation ROperation.new("empty",type("bool"))
            klass.add_operation ROperation.new("reserve",type("void"),RParameter.new("size",type("size_t")))
            klass.add_operation ROperation.new("operator[]",vector_type,RParameter.new("size",type("size_t")))
            klass.add_operation ROperation.new("at",vector_type,RParameter.new("size",type("size_t")))
            klass.add_operation ROperation.new("front",vector_type)
            klass.add_operation ROperation.new("back",vector_type)
            klass.add_operation ROperation.new("data",type("void *"))
            klass.add_operation ROperation.new("push_back",type("void"),RParameter.new("other",vector_type).to_const)
            klass.add_operation ROperation.new("pop_back",type("void"))
            klass.add_operation ROperation.new("swap",type("void"),RParameter.new("other",klass))
            # add ruby code to the front of the method
            klass.operation("operator[]").specialize_ruby do
                "validate_index(size)"
            end
            klass.operation("at").specialize_ruby do
                "validate_index(size)"
            end

            specialize_ruby do
    %Q$     def self.new(type,*args)
            klass,elements = if type.class == Class
                                [type.name,[]]
                            else
                                e = Array(type) + args.flatten
                                args = []
                                [type.class.name,e]
                            end
            #remove module name
            klass = klass.split("::")
            klass.shift if klass.size > 1
            klass = klass.join("_")
            raise ArgumentError,"no std::vector defined for \#{type}" unless self.const_defined?(klass)
            v = self.const_get(klass).new(*args)
            elements.each do |e|
                v << e
            end
            v
            end$
            end

            klass
        end

        # called from RTemplate when ruby_specialize is called for the instance
        def specialize_ruby_specialization(klass)
            %Q$ include Enumerable
            alias get_element []
            def [](idx)
                validate_index(idx)
                get_element(idx)
            end

            def validate_index(idx)
                if idx < 0 || idx >= size
                    raise RangeError,"\#{idx} is out of range [0..\#{size-1}]"
                end
            end
            def each(&block)
                if block
                     s = size
                     0.upto(s-1) do |i|
                         yield self[i]
                     end
                else
                    Enumerator.new(self)
                end
            end
            def <<(val)
                push_back(val)
                self
            end
            def delete_if(&block)
                v = self.class.new
                each do |i|
                     v << i if !yield(i)
                end
                v.swap(self)
                self
            end$
        end
    end
end
