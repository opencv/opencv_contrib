module Rbind
    class StdString < RClass
        def initialize(name,root)
            super(name)

            size_t = root.type("size_t")
            char = root.type("char")
            string = root.type("c_string")
            const_string = string.to_const
            bool = root.type("bool")
            void = root.type("void")
            int = root.type("int")
            
            add_operation ROperation.new(self.name,nil)
            add_operation ROperation.new(self.name,nil,RParameter.new("other",self).to_const)
            add_operation ROperation.new(self.name,nil,RParameter.new("str",string).to_const,RParameter.new("size",size_t))
            add_operation ROperation.new("size",size_t)
            add_operation ROperation.new("length",size_t)
            add_operation ROperation.new("operator[]",char,RParameter.new("idx",size_t))
            add_operation ROperation.new("c_str",const_string)
            add_operation ROperation.new("empty",bool)
            add_operation ROperation.new("clear",void)
            add_operation ROperation.new("compare",int,RParameter.new("other",self).to_const)
            add_operation ROperation.new("swap",void,RParameter.new("other",self))

            specialize_ruby do
%Q$         def self.to_native(obj,context)
                if obj.is_a? ::String
                    str = obj.to_str
                    Std::String.new(str,str.length).__obj_ptr__
                else
                    rbind_to_native(obj,context)
                end
            end
            def to_s
                c_str
            end$
            end
        end
    end
end
