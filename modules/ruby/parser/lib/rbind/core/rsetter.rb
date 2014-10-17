
module Rbind
    class RSetter < ROperation
        attr_reader :attribute

        def initialize(attr)
            @attribute = attr
            para = RParameter.new("value",attr.type)
            super("set_#{attr.name}",RDataType.new("void"),para)
        end

        def attribute?
            true
        end

        def signature
            attribute.signature
        end
    end
end
