
module Rbind
    class REnum < RDataType
        attr_accessor :values

        def initialize(name)
            super(name)
            @values = Hash.new
        end

        def generate_signatures
            ["#{full_name}","#{cname}"]
        end

        def basic_type?
            true
        end

        def add_value(name,val)
            @values[name] = val
        end
    end
end
