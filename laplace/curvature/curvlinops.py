from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch
from curvlinops import (
    EFLinearOperator,
    FisherMCLinearOperator,
    FisherType,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
    KFACLinearOperator_CrossVersion
)
from curvlinops._base import _LinearOperator
from torch import nn

from laplace.curvature import CurvatureInterface, EFInterface, GGNInterface
from laplace.utils import Kron, Likelihood


class CurvlinopsInterface(CurvatureInterface):
    """Interface for Curvlinops backend. <https://github.com/f-dangel/curvlinops>"""

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
    ) -> None:
        super().__init__(
            model, likelihood, last_layer, subnetwork_indices, dict_key_x, dict_key_y
        )

    @property
    def _kron_fisher_type(self) -> str:
        raise NotImplementedError

    @property
    def _linop_context(self) -> type[_LinearOperator]:
        raise NotImplementedError

    @staticmethod
    def _rescale_kron_factors(kron: Kron, M: int, N: int) -> Kron:
        # Renormalize Kronecker factor to sum up correctly over N data points with
        # batches of M. For M=N (full-batch) just M/N=1
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= M / N
        return kron

    def _get_kron_factors(self, linop: KFACLinearOperator) -> Kron:
        kfacs = list()
        names_ = []
        for name, module in self.model.named_modules():
            if name not in linop._mapping.keys():
                continue

            A = linop._input_covariances[name] #this is matrix A
            B = linop._gradient_covariances[name] #this is matrix G
            print('_get_kron_factors', A.dtype, B.dtype)

            if hasattr(module, "bias") and module.bias is not None:
                kfacs.append([B, A])
                kfacs.append([B])
                names_.append(name+'.bias')
            elif hasattr(module, "weight"):
                p, q = B.numel(), A.numel()
                if p == q == 1:
                    kfacs.append([B * A])
                else:
                    kfacs.append([B, A])
                names_.append(name + '.weight')
            else:
                raise ValueError(f"Whats happening with {module}?")
        return Kron(kfacs), names_

    def _get_kron_factors_cross(self, linop: KFACLinearOperator) -> Kron:
        kfacs = list()
        names_ = []
        for name, module in self.model.named_modules():
            if name not in list(linop._mapping.keys())[::2]:
                continue
            names_.append(name)
            A = linop._cross_input_covariances[name]  # this is matrix A
            B = linop._cross_gradient_covariances[name]  # this is matrix G

            if hasattr(module, "bias") and module.bias is not None:
                kfacs.append([B, A])
                kfacs.append([B])
            elif hasattr(module, "weight"):
                p, q = B.numel(), A.numel()
                if p == q == 1:
                    kfacs.append([B * A])
                else:
                    kfacs.append([B, A])
            else:
                raise ValueError(f"Whats happening with {module}?")
        return Kron(kfacs), names_

    def kron(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        N: int,
        hessian_str: str = "kron",
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, Kron]:
        if isinstance(x, (dict, MutableMapping)):
            kwargs["batch_size_fn"] = lambda x: x[self.dict_key_x].shape[0]

        if hessian_str == 'kron':
            linop = KFACLinearOperator(
                self.model,
                self.lossfunc,
                self.params,
                [(x, y)],
                fisher_type=self._kron_fisher_type,
                separate_weight_and_bias=True,
                check_deterministic=False,  # To avoid overhead
                # `kwargs` for `mc_samples` when `stochastic=True` and `kfac_approx` to
                # choose between `'expand'` and `'reduce'`.
                # Defaults to `mc_samples=1` and `kfac_approx='expand'.
                **kwargs,
            )
            print('Linop',linop)
            linop._compute_kfac()

            kron, names_ = self._get_kron_factors(linop)
            print('kron', kron.kfacs[0][0][0].dtype)
            kron = self._rescale_kron_factors(kron, len(y), N)
            print('kron', kron.kfacs[0][0][0].dtype)
            kron *= self.factor
            print('kron', kron.kfacs[0][0][0].dtype)
            device_used = next(self.model.parameters()).device
            y = y[:, 0].to(device_used, non_blocking=True) #TODO THIS IS MINE
            x = {key: value.to(device_used, non_blocking=True) for key, value in x.items()} #TODO THIS IS MINE
            loss = self.lossfunc(self.model(x), y)
            #names_order = list(self.params_dict.keys())
            sizes_of_layers = [
                j.shape
                for j in self.params
            ]

            return self.factor * loss.detach(), kron, names_, sizes_of_layers #TODO HERE CAHNGE

        elif hessian_str == 'kron-cross':
            linop = KFACLinearOperator_CrossVersion(
                self.model,
                self.lossfunc,
                self.params,
                [(x, y)],
                fisher_type=self._kron_fisher_type,
                separate_weight_and_bias=True,
                names_list=list(self.params_dict.keys()),
                check_deterministic=False,  # To avoid overhead
                # `kwargs` for `mc_samples` when `stochastic=True` and `kfac_approx` to
                # choose between `'expand'` and `'reduce'`.
                # Defaults to `mc_samples=1` and `kfac_approx='expand'.
                **kwargs,
            )

            linop._compute_kfac()

            kron, names_ = self._get_kron_factors(linop)
            kronc, names2_ = self._get_kron_factors_cross(linop)
            kron = self._rescale_kron_factors(kron, len(y), N)
            kronc = self._rescale_kron_factors(kronc, len(y), N)
            kron *= self.factor
            kronc *= self.factor
            device_used = next(self.model.parameters()).device
            y = y[:, 0].to(device_used, non_blocking=True) #TODO THIS IS MINE
            x = {key: value.to(device_used, non_blocking=True) for key, value in x.items()} #TODO THIS IS MINE
            loss = self.lossfunc(self.model(x), y)
            names_order = list(self.params_dict.keys())
            sizes_of_layers = [
                j.shape
                for j in self.params
            ]

            return self.factor * loss.detach(), (kron,kronc), names_order, sizes_of_layers #TODO HERE CAHNGE


    def full(
        self,
        x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        y: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fallback to torch.func backend for SubnetLaplace
        if self.subnetwork_indices is not None:
            return super().full(x, y, **kwargs)

        curvlinops_kwargs = {k: v for k, v in kwargs.items() if k != "N"}
        if isinstance(x, (dict, MutableMapping)):
            curvlinops_kwargs["batch_size_fn"] = lambda x: x[self.dict_key_x].shape[0]

        linop = self._linop_context(
            self.model,
            self.lossfunc,
            self.params,
            [(x, y)],
            check_deterministic=False,
            **curvlinops_kwargs,
        )
        H = torch.as_tensor(
            linop @ torch.eye(linop.shape[0]),
            device=next(self.model.parameters()).device,
        )

        f = self.model(x)
        loss = self.lossfunc(f, y)

        return self.factor * loss.detach(), self.factor * H


class CurvlinopsGGN(CurvlinopsInterface, GGNInterface):
    """Implementation of the `GGNInterface` using Curvlinops."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: Likelihood | str,
        last_layer: bool = False,
        subnetwork_indices: torch.LongTensor | None = None,
        dict_key_x: str = "input_ids",
        dict_key_y: str = "labels",
        stochastic: bool = False,
    ) -> None:
        super().__init__(
            model, likelihood, last_layer, subnetwork_indices, dict_key_x, dict_key_y
        )
        self.stochastic = stochastic

    @property
    def _kron_fisher_type(self) -> FisherType:
        return FisherType.MC if self.stochastic else FisherType.TYPE2

    @property
    def _linop_context(self) -> type[_LinearOperator]:
        return FisherMCLinearOperator if self.stochastic else GGNLinearOperator


class CurvlinopsEF(CurvlinopsInterface, EFInterface):
    """Implementation of `EFInterface` using Curvlinops."""

    @property
    def _kron_fisher_type(self) -> FisherType:
        return FisherType.EMPIRICAL

    @property
    def _linop_context(self) -> type[_LinearOperator]:
        return EFLinearOperator


class CurvlinopsHessian(CurvlinopsInterface):
    """Implementation of the full Hessian using Curvlinops."""

    @property
    def _linop_context(self) -> type[_LinearOperator]:
        return HessianLinearOperator
