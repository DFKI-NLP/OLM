from typing import Dict, Tuple, List, Optional


class TokenField:
    def __init__(self, tokens: List[str]) -> None:
        self._tokens = tokens

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    def __repr__(self) -> str:
        return f"TokenField({self.tokens})"


class OccludedTokenField(TokenField):
    def __init__(self,
                 tokens: List[str],
                 occluded_index: int,
                 occlude_token: str) -> None:
        super().__init__(tokens)
        self._occluded_index = occluded_index
        self._occlude_token = occlude_token

    @property
    def tokens(self) -> List[str]:
        tmp_tokens = list(self._tokens)
        tmp_tokens[self._occluded_index] = self._occlude_token
        return tmp_tokens

    @property
    def occluded_index(self) -> int:
        return self._occluded_index

    def __repr__(self) -> str:
        return f"OccludedTokenField({self.tokens})"

    @classmethod
    def from_token_field(cls,
                         token_field: TokenField,
                         occluded_index: int,
                         occlude_token: str) -> "OccludedTokenField":
        return cls(token_field.tokens, occluded_index, occlude_token)


class DeletedTokenField(OccludedTokenField):
    def __init__(self,
                 tokens: List[str],
                 occluded_index: int) -> None:
        super().__init__(tokens, occluded_index, occlude_token="")
        self._occluded_index = occluded_index

    @property
    def tokens(self) -> List[str]:
        tmp_tokens = list(self._tokens)
        del tmp_tokens[self._occluded_index]
        return tmp_tokens

    def __repr__(self) -> str:
        return f"DeletedTokenField({self.tokens})"

    @classmethod
    def from_token_field(cls,
                         token_field: TokenField,
                         occluded_index: int) -> "DeletedTokenField":
        return cls(token_field.tokens, occluded_index)


class InputInstance:
    def __init__(self,
                 id_: str,
                 **token_fields: Dict[str, List[str]]) -> None:
        self.id = id_
        self.token_fields = {name: TokenField(tokens)
                             for name, tokens in token_fields.items()}
        for key, value in self.token_fields.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"InputInstance(id={self.id}, token_fields={self.token_fields})"


class OccludedInstance:
    def __init__(self,
                 id_: str,
                 token_fields: Dict[str, TokenField],
                 weight: float = 1.) -> None:
        self.id = id_
        self.weight = weight
        self.token_fields = token_fields
        for key, value in token_fields.items():
            setattr(self, key, value)

    @property
    def occluded_indices(self) -> Optional[Tuple[str, int]]:
        indices = []
        for name, field in self.token_fields.items():
            if isinstance(field, OccludedTokenField):
                indices.append((name, field.occluded_index))

        # for now, only allow up to one occluded token per instance
        assert len(indices) <= 1

        return indices[0] if indices else None

    def __repr__(self) -> str:
        return f"OccludedInstance(id={self.id}, token_fields={self.token_fields}), weight={self.weight})"

    @classmethod
    def from_input_instance(cls,
                            input_instance: InputInstance,
                            occlude_token: Optional[str] = None,
                            occlude_field_index: Optional[Tuple[str, int]] = None,
                            weight: float = 1.) -> "OccludedInstance":

        if occlude_token is not None and occlude_field_index is None:
            raise ValueError("'occlude_token' requires setting 'occlude_field_index'.")

        token_fields = input_instance.token_fields

        if occlude_field_index is not None:
            token_fields = dict(token_fields)
            field_name, field_index = occlude_field_index
            token_field = token_fields[field_name]

            if occlude_token is None:
                occluded_token_field = DeletedTokenField.from_token_field(token_field,
                                                                          occluded_index=field_index)
            else:
                occluded_token_field = OccludedTokenField.from_token_field(token_field,
                                                                           occluded_index=field_index,
                                                                           occlude_token=occlude_token)
            token_fields[field_name] = occluded_token_field

        return cls(id_=input_instance.id,
                   token_fields=token_fields,
                   weight=weight)
