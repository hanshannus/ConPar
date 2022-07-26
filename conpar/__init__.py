import io
import functools
from pathlib import Path
from importlib import import_module
from urllib.parse import urlparse
from typing import Callable, Any, Union, List, Dict


def _is_local_file(string: str) -> bool:
    """Check if input points to a local file.

    Parameters
    ----------
    string : str
        Value to check.

    Returns
    -------
    bool
        True if file exists.
    """
    if string[:2] == "~/":
        string = Path.home() / string[2:]
    return Path(string).exists()


def _is_url_s3(string: str) -> bool:
    """Check if input is valid S3 URL.
    Parameters
    ----------
    string : str
        Value to check.

    Returns
    -------
    bool
        True if input is valid S3 URL.
    """
    return urlparse(string, allow_fragments=False).scheme == "s3"


def _is_url_blob(string: str) -> bool:
    """Check if input is valid Azure Blob URL.

    Parameters
    ----------
    string : str
        Value to check.

    Returns
    -------
    bool
        True if input is valid Azure Blob URL.
    """
    scheme = urlparse(string, allow_fragments=False).scheme
    res = urlparse(string, allow_fragments=False).netloc.split(".")
    if len(res) < 2:
        return False
    return scheme == "s3" and res[1] == "blob"


def _read_file_s3(
    uri: Union[str, Path],
    **kwargs,
) -> str:
    """Download and read file from S3 storage.

    Parameters
    ----------
    uri : Union[str, Path]
        URL pointing to file.

    Returns
    -------
    str
        Content of blob file.

    Raises
    ------
    ValueError
        Invalid URL.
    """
    import boto3

    # parse url
    parse = urlparse(uri, allow_fragments=False)
    if not parse.scheme:
        raise ValueError(f"Input '{uri}' is no valid URL.")
    bucket_name = parse.netloc
    file_path = parse.path.lstrip("/")
    # start a Boto3 session to load configurations
    boto3_session = boto3.Session(**kwargs)
    # get S3 instance
    s3 = boto3_session.resource("s3")
    # search for object
    s3_obj = s3.Object(bucket_name, file_path).get()
    # return content of S3 object
    return s3_obj["Body"].read().decode("utf-8")


def _read_file_blob(
    url: Union[str, Path],
    **kwargs,
) -> str:
    """Download and read file from Azure blob storage.

    Parameters
    ----------
    url : Union[str, Path]
        URL pointing to file.

    Returns
    -------
    str
        Content of blob file.
    """
    from azure.storage.blob import BlobClient

    blob = BlobClient.from_blob_url(blob_url=url, **kwargs)

    return blob.download_blob().readall().decode("utf-8")


def _find_loader(string: Union[str, Path]) -> Callable:
    """Infer data loader from input suffix.

    Supported file extensions:
        - json
        - yaml

    Parameters
    ----------
    string : Union[str, Path]
        Path to configuration file.

    Returns
    -------
    Callable
        Dataloader for JSON or YAML.

    Raises
    ------
    NotImplementedError
        Only JSON and YAML files are supported.
    """
    file_path = Path(string)
    suffix = file_path.suffix[1:].lower()
    if suffix in ["json"]:
        from json import load
    elif suffix in ["yml", "yaml"]:
        from yaml import load
    else:
        raise NotImplementedError(f"file extension {file_path.suffix} not known")
    return load


def input_type_handler(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(body: Any, *args, **kwargs) -> Any:
        # parse body
        if isinstance(body, (list, tuple)):
            return [func(i, *args, **kwargs) for i in body]
        elif isinstance(body, dict):
            for key, value in body.items():
                body[key] = func(value, *args, **kwargs)
            return body
        return func(body, *args, **kwargs)

    return wrapper


@input_type_handler
def _parse_import_with_eval(
    body: Union[str, List[str], Dict[str, str]],
    header: Union[str, List[str], Dict[str, str]] = None,
) -> Any:
    """Evaluate input by utilizing headers.

    Parameters
    ----------
    body : Union[str, List[str], Dict[str, str]]
        Input to evaluate.
    header : Union[str, List[str], Dict[str, str]], optional
        Preparations to be done before evaluation, by default None

    Returns
    -------
    Any
        Result of evaluation.
    """
    # condense header
    if isinstance(header, (list, tuple)):
        header = "\n".join(header) + "\n"
    # prepend header so it is executed before body
    call = header if header is not None else ""
    # write result of 'body' evaluation into (local) 'res' value
    call += f"res = {body}"
    # define dict to return local values from 'exec'
    loc = {}
    # execute call
    exec(call, globals(), loc)
    # retrieve 'res" value from local dict
    return loc["res"]


@input_type_handler
def _parse_eval(
    body: Union[str, List[str], Dict[str, str]],
) -> Any:
    """Evaluate input.

    Function is decorated to handle list, tuple and dict input differently.

    Parameters
    ----------
    body : str
        String from which to evaluate.

    Returns
    -------
    Any
        Result of evaluation
    """
    # evaluate input
    return eval(body)


@input_type_handler
def _parse_import(
    body: str,
) -> Any:
    """Import modules from input.

    Function is decorated to handle list, tuple and dict input differently.

    Parameters
    ----------
    body : str
        String from which to import a module.

    Returns
    -------
    Any
        Imported module.

    Raises
    ------
    ModuleNotFoundError
        Requested module cannot be found.
    ImportError
        Invalid import format. Supported formats: 'from ... import ...' and
        'import ...'
    """
    # only 'from ... import ...' or 'import ...' are valid inputs
    tmp = body.split()
    if len(tmp) == 4 and tmp[0] == "from" and tmp[2] == "import":
        # case 'from ... import ...'
        body = getattr(import_module(tmp[1]), tmp[3])
    elif len(tmp) == 2 and tmp[0] == "import":
        # case 'import ...'
        try:
            body = import_module(tmp[1])
        except ModuleNotFoundError:
            # fall back to case 'from ... import ...'
            res = tmp[1].rsplit(".")
            if len(res) != 2:
                raise ModuleNotFoundError
            body = getattr(import_module(res[0]), res[1])
    else:
        raise ImportError("Supported formats:\nfrom ... import ...\nimport ...")
    return body


@input_type_handler
def _parse_load(string: str, **kwargs) -> dict:
    """Load content of file or URL.

    Parameters
    ----------
    string : str
        Text or path to file (local, AWS S3, Azure blob)
    **kwargs
        Keywords for AWS (boto3.Session) or Azure (BlobClient.from_blob_url)

    Returns
    -------
    dict
        Content of file.

    Raises
    ------
    FileNotFoundError
        Input file cannot be found.
    """
    if _is_url_s3(string):
        # found S3 scheme
        text = _read_file_s3(string, **kwargs)
    elif _is_url_blob(string):
        # found Blob scheme
        text = _read_file_blob(string, **kwargs)
    elif _is_local_file(string):
        # found local file
        with open(string, "r") as fp:
            text = fp.read()
    else:
        raise FileNotFoundError(f"Cannot locate {string}.")
    # acquire data loader based on suffix of file
    load = _find_loader(string)
    return load(io.StringIO(text))


def _parse_dict(cfg: dict) -> Union[Any, dict]:
    """Parse configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    Union[Any, dict]
        Parsed configuration dictionary.
    """
    # handle key words
    if "_eval_" in cfg.keys() and "_import_" in cfg.keys():
        obj = _parse_import_with_eval(cfg.pop("_eval_"), cfg.pop("_import_"))
    elif "_eval_" in cfg.keys():
        obj = _parse_eval(cfg.pop("_eval_"))
    elif "_import_" in cfg.keys():
        obj = _parse_import(cfg.pop("_import_"))
    elif "_load_" in cfg.keys():
        loaded_cfg = _parse_load(cfg.pop("_load_"))
        # need to start over for loaded config
        obj = parse(loaded_cfg)
    else:
        obj = cfg
    # condense output
    if len(cfg.items()) == 0:
        cfg = obj
    else:
        cfg["object"] = obj
    return cfg


def parse(cfg: Any) -> Union[Any, dict]:
    """Parse configuration files with special keywords.

    Valid keywords are:
        - '_load_': parse the config file path and return the content
        - '_import_': import python packages
        - '_eval_': Evaluate value utilizing imported packages

    Parameters
    ----------
    cfg : Any
        List or dict to parse.

    Returns
    -------
    Union[Any, dict]
        [description]
    """
    if isinstance(cfg, (list, tuple)):
        cfg = [parse(i) for i in cfg]
    elif isinstance(cfg, dict):
        cfg = _parse_dict(cfg)

    return cfg
