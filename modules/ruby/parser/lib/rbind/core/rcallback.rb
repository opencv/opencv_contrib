
module Rbind
    class RCallback < RDataType
        attr_reader :fct

        def initialize(name,return_type,*args)
            @fct = ROperation.new(name,return_type,*args)
            super(name)
        end

        def callback?
            true
        end
    end
end
