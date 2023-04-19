#!/usr/bin/env python3
import csv

DEFAULT_INTERFACE_TEMPLATE_PATH = 'multi-query-constrained-hit.html'

HEADER = '<html><head></head><body><form id="mturk_form" name="mturk_form">'
ALERT_OK = (
    "$('#alerts')"
    "    .attr('class', 'alert alert-success')"
    "    .append($('<div>')"
    "        .text('In production, your solution would now be submitted:'))"
    "    .append($('<pre>')"
    "        .append($('<code>')"
    "            .text(JSON.stringify(output, null, 2))));"
    "return false;"
)
FOOTER = (
    '<div class="text-center">'
    '<input id="submitButton" class="btn btn-primary" type="submit" value="Submit" />'
    '</div></form></body></html>'
)


def main():
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser(description='Make static HTML file to demo interface with single HIT')
    parser.add_argument('--interface-template-path', help='Path to interface HTML template', default=DEFAULT_INTERFACE_TEMPLATE_PATH)
    parser.add_argument('--sampler', action='store_true', help='Sample a line from data file on each page load')
    parser.add_argument('--no-submit', action='store_true', help='Pressing submit button just shows output data')
    parser.add_argument('toy_data_path', help='Path to single JSON-encoded HIT data')
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    with open(args.interface_template_path) as f:
        interface_template = f.read()
    with open(args.toy_data_path) as f:
        if args.sampler:
            toy_data = '[' + ','.join(line.strip() for line in f.readlines()) + ']'
            toy_data_code = '_.sample({})'.format(toy_data)
        else:
            toy_data_code = f.read()

    interface_body = interface_template.replace('${json_data}', toy_data_code)
    if args.no_submit or not args.sampler:
        interface_body = interface_body.replace("$('#alerts').attr('class', '').text('');", ALERT_OK)

    print(HEADER + '\n\n' + interface_body + '\n' + FOOTER)


if __name__ == '__main__':
    main()
