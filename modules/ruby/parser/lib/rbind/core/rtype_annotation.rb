require 'delegate'

module Rbind
    class RTypeAnnotation < SimpleDelegator
        attr_accessor :const

        def initialize(type,options=Hash.new)
            super(type)
            @options = options
        end

        def to_annotation(key,val)
            if @options.has_key?(key) && @options[key] == val
                self
            else
                opt = @options.clone
                opt[key] = val
                RTypeAnnotation.new(__getobj__,opt)
            end
        end

        def remove_annotation(key)
            opt = @options.clone
            opt.delete(key)
            if opt.empty?
                __getobj__
            else
                RTypeAnnotation.new(__getobj__,opt)
            end
        end

        def const?
            if @options.has_key? :const
                !!@options[:const]
            else
                super
            end
        end

        def to_const
            to_annotation(:const,true)
        end

        def remove_const
            remove_annotation(:const)
        end

        def ownership?
            if @options.has_key? :ownership
                !!@options[:ownership]
            else
                super
            end
        end

        def to_ownership(val=true)
            if ptr?
                to_annotation(:ownership,val)
            else
                super
            end
        end

        def remove_ownership
            remove_annotation(:ownership)
        end

        def to_ptr
            val = if ptr?
                      @options[:ptr]+1
                    else
                        1
                    end
            to_annotation(:ptr,val)
        end

        def ptr?
            if @options.has_key? :ptr
                @options[:ptr] != 0
            else
                super
            end
        end

        def remove_ptr
            return self if !ptr?
            val = @options[:ptr]-1
            if val == 0
                remove_annotation(:ptr)
            else
                to_annotation(:ptr,val)
            end
        end

        def to_ref
            to_annotation? :ref,true
        end

        def ref?
            if @options.has_key? :ref
                !!@options[:ref]
            else
                super
            end
        end

        def remove_ref
            remove_annotation :ref
        end

        def raw?
            false
        end

        def signature(sig=nil)
            generate_signatures[0]
        end

        def csignature(sig=nil)
            generate_signatures[1]
        end

        def generate_signatures
            str_pre = ""
            str_post = ""

            str_pre = "const " if const?
            str_post += "&" if ref?
            str_post += "*" * @options[:ptr] if ptr?
            __getobj__.generate_signatures.map do |s|
                str_pre + s + str_post
            end
        end
    end
end
