
module Rbind
    class RParameter < RAttribute
        attr_accessor :default_value

        def initialize(name,type,default_value=nil)
            super(name,type)
            self.default_value = default_value
        end

        def default_value=(val)
            val = val.to_s
            @default_value = if !val.empty?
                                 val.chomp.chomp(" ")
                             else
                                 nil
                             end
        end

        def default_value(val = nil)
            if val
                self.default_value = val
                self
            else
                @default_value
            end
        end

        def to_single_ptr
            t = self.clone
            t.type = type.to_single_ptr
            t
        end

        def remove_const!
            @type = type.remove_const
            self
        end

        def const!
            return self if const?
            @type = type.to_const
            self
        end

        def to_const
            return self if const?
            para = self.dup
            para.type = type.to_const
            self
        end

        def const?
            type.const?
        end

        def ref?
            type.ref?
        end

        def basic_type?
            type.basic_type?
        end

        def generate_signatures
            if default_value
                sigs = super
                sigs[0] += " = #{default_value}"
                sigs
            else
                super
            end
        end
    end
end
