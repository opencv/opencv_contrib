module Rbind
    class StdMap < RTemplateClass
        def specialize(klass,*parameters)
            if parameters.size < 2
                raise ArgumentError,"StdMap does require at least two template parameters. Got: #{parameters}}"
            end
            map_key_type = parameters.flatten[0]
            map_value_type = parameters.flatten[1]
            if parameters.size > 2
                map_comp_type = parameters.flatten[2]
            else
                map_comp_type = nil
            end

            klass.add_operation ROperation.new(klass.name,nil)
            klass.add_operation ROperation.new(klass.name,nil,RParameter.new("other",klass).to_const)

            klass.add_operation ROperation.new("size",type("size_t"))
            klass.add_operation ROperation.new("clear",type("void"))
            klass.add_operation ROperation.new("capacity",type("size_t"))
            klass.add_operation ROperation.new("empty",type("bool"))
            klass.add_operation ROperation.new("operator[]",map_value_type, RParameter.new("key_type", map_key_type))
            klass.add_operation ROperation.new("at",map_value_type, RParameter.new("key_type",map_key_type))
            klass.add_operation ROperation.new("erase",type("void"), RParameter.new("key_type",map_key_type))

            klass
        end
    end
end
